use std::fs;
use std::path::{Path, PathBuf};
use std::io::{self, Write, Read};
use std::num::NonZeroU32;
use serde::{Deserialize, Serialize};
use clap::Parser;
use anyhow::{Context, Result};
use std::collections::HashMap;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token_type::LlamaTokenAttr;
use llama_cpp_2::LogOptions;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model tier to run: speed, fast, balanced
    #[arg(short, long)]
    tier: Option<String>,

    /// Custom prompt to run instead of interactive chat
    #[arg(short, long)]
    prompt: Option<String>,

    /// Suppress all initialization logs and only print assistant output
    #[arg(long)]
    output_only: bool,

    /// Chat history JSON file path for continuous chat
    #[arg(long)]
    chat: Option<String>,

    /// List all available model tiers and exit
    #[arg(long)]
    list_tiers: bool,

    /// Override temperature sampler parameter
    #[arg(long)]
    temp: Option<f32>,

    /// Override top_p sampler parameter
    #[arg(long)]
    top_p: Option<f32>,

    /// Override context window size (n_ctx)
    #[arg(long)]
    ctx: Option<u32>,

    /// Override GPU offloaded layers count (n_gpu_layers)
    #[arg(long)]
    gpu_layers: Option<i32>,

    /// Enable printing of model reasoning/thinking steps to the console
    #[arg(long)]
    show_thinking: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelTier {
    name: String,
    url: String,
    filename: String,
    n_gpu_layers: i32,
    template: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct InferenceConfig {
    n_ctx: u32,
    temp: f32,
    top_p: f32,
    use_mmap: bool,
    use_mlock: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Config {
    default_tier: String,
    memory_limit_mb: u32,
    tiers: HashMap<String, ModelTier>,
    inference: InferenceConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn get_binary_dir() -> Result<PathBuf> {
    let mut exe_path = std::env::current_exe()?;
    exe_path.pop(); // Remove binary name to get directory
    Ok(exe_path)
}

fn load_config(bin_dir: &Path) -> Result<Config> {
    let config_path = bin_dir.join("config.json");
    if !config_path.exists() {
        let default_config = include_str!("../config.json");
        fs::write(&config_path, default_config)?;
    }
    let content = fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&content)?;
    Ok(config)
}

fn download_model(url: &str, dest: &Path) -> Result<()> {
    let tmp_dest = dest.with_extension("tmp");
    println!("Downloading model from {} to {}...", url, dest.display());
    
    let download_res = (|| -> Result<()> {
        let mut resp = reqwest::blocking::get(url)?;
        if !resp.status().is_success() {
            anyhow::bail!("Failed to download model: HTTP {}", resp.status());
        }
        let total_size = resp
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|ct_len| ct_len.to_str().ok())
            .and_then(|ct_len_str| ct_len_str.parse::<u64>().ok())
            .unwrap_or(0);

        let pb = indicatif::ProgressBar::new(total_size);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );

        let mut file = fs::File::create(&tmp_dest)?;
        let mut buffer = vec![0; 8192];
        loop {
            let n = resp.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n])?;
            pb.inc(n as u64);
        }
        pb.finish_with_message("Download complete");
        Ok(())
    })();

    if download_res.is_ok() {
        fs::rename(&tmp_dest, dest)?;
        Ok(())
    } else {
        if tmp_dest.exists() {
            let _ = fs::remove_file(&tmp_dest);
        }
        download_res
    }
}

fn token_to_string(model: &LlamaModel, token: LlamaToken) -> String {
    match model.token_to_piece_bytes(token, 32, false, None) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => {
            if let Ok(bytes) = model.token_to_piece_bytes(token, 256, false, None) {
                String::from_utf8_lossy(&bytes).into_owned()
            } else {
                String::new()
            }
        }
    }
}

fn format_prompt(template: &str, system: &str, user: &str) -> String {
    match template {
        "qwen" => format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        ),
        "llama" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            system, user
        ),
        _ => format!("System: {}\nUser: {}\nAssistant: ", system, user),
    }
}

fn format_chat_history(template: &str, default_system: &str, messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let (system_content, chat_messages) = if !messages.is_empty() && messages[0].role == "system" {
        (messages[0].content.as_str(), &messages[1..])
    } else {
        (default_system, messages)
    };

    match template {
        "qwen" => {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", system_content));
            for msg in chat_messages {
                prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
            }
            prompt.push_str("<|im_start|>assistant\n");
        }
        "llama" => {
            prompt.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
            prompt.push_str(system_content);
            prompt.push_str("<|eot_id|>");
            for msg in chat_messages {
                prompt.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>", role=msg.role, content=msg.content));
            }
            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        _ => {
            prompt.push_str(&format!("System: {}\n", system_content));
            for msg in chat_messages {
                prompt.push_str(&format!("{}: {}\n", if msg.role == "user" { "User" } else { "Assistant" }, msg.content));
            }
            prompt.push_str("Assistant: ");
        }
    }
    prompt
}

fn load_chat_history(path: &Path) -> Result<Vec<ChatMessage>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(path)?;
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }
    let messages: Vec<ChatMessage> = serde_json::from_str(&content)?;
    Ok(messages)
}

fn save_chat_history(path: &Path, messages: &[ChatMessage]) -> Result<()> {
    let content = serde_json::to_string_pretty(messages)?;
    fs::write(path, content)?;
    Ok(())
}

struct ThinkingFilter {
    show_thinking: bool,
    in_thinking: bool,
    buffer: String,
}

impl ThinkingFilter {
    fn new(show_thinking: bool) -> Self {
        Self {
            show_thinking,
            in_thinking: false,
            buffer: String::new(),
        }
    }

    fn process(&mut self, piece: &str) -> String {
        self.buffer.push_str(piece);
        let mut output = String::new();

        loop {
            if !self.in_thinking {
                if let Some(pos) = self.buffer.find("<think>") {
                    output.push_str(&self.buffer[..pos]);
                    self.buffer = self.buffer[pos + 7..].to_string();
                    self.in_thinking = true;
                    if self.show_thinking {
                        output.push_str("<think>");
                    }
                } else if let Some(pos) = self.buffer.find("<thinking>") {
                    output.push_str(&self.buffer[..pos]);
                    self.buffer = self.buffer[pos + 10..].to_string();
                    self.in_thinking = true;
                    if self.show_thinking {
                        output.push_str("<thinking>");
                    }
                } else {
                    let keep_len = 10.min(self.buffer.len());
                    let split_pos = self.buffer.len() - keep_len;
                    output.push_str(&self.buffer[..split_pos]);
                    self.buffer = self.buffer[split_pos..].to_string();
                    break;
                }
            } else {
                if let Some(pos) = self.buffer.find("</think>") {
                    if self.show_thinking {
                        output.push_str(&self.buffer[..pos + 8]);
                    }
                    self.buffer = self.buffer[pos + 8..].to_string();
                    self.in_thinking = false;
                } else if let Some(pos) = self.buffer.find("</thinking>") {
                    if self.show_thinking {
                        output.push_str(&self.buffer[..pos + 11]);
                    }
                    self.buffer = self.buffer[pos + 11..].to_string();
                    self.in_thinking = false;
                } else {
                    if self.show_thinking {
                        let keep_len = 11.min(self.buffer.len());
                        let split_pos = self.buffer.len() - keep_len;
                        output.push_str(&self.buffer[..split_pos]);
                        self.buffer = self.buffer[split_pos..].to_string();
                    } else {
                        let keep_len = 11.min(self.buffer.len());
                        let split_pos = self.buffer.len() - keep_len;
                        self.buffer = self.buffer[split_pos..].to_string();
                    }
                    break;
                }
            }
        }
        output
    }

    fn flush(&mut self) -> String {
        let res = if !self.in_thinking || self.show_thinking {
            self.buffer.clone()
        } else {
            String::new()
        };
        self.buffer.clear();
        res
    }
}

fn run_inference(
    model: &LlamaModel,
    backend: &LlamaBackend,
    config: &Config,
    _tier: &ModelTier,
    prompt_text: &str,
    show_thinking: bool,
) -> Result<String> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.inference.n_ctx));
    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("Failed to create context")?;

    let tokens = model
        .str_to_token(prompt_text, AddBos::Always)
        .context("Failed to tokenize prompt")?;

    let mut batch = LlamaBatch::new(config.inference.n_ctx as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i == tokens.len() - 1;
        batch.add(*token, i as i32, &[0], is_last)?;
    }

    ctx.decode(&mut batch).context("Failed to decode prompt")?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(config.inference.temp),
        LlamaSampler::top_p(config.inference.top_p, 1),
        LlamaSampler::greedy(),
    ]);

    let mut n_cur = tokens.len() as i32;
    let mut token = sampler.sample(&ctx, batch.n_tokens() - 1);
    sampler.accept(token);

    let mut filter = ThinkingFilter::new(show_thinking);
    let mut generated_text = String::new();

    if !model.token_attr(token).contains(LlamaTokenAttr::Control) {
        let first_piece = token_to_string(model, token);
        if !first_piece.contains("<|im_end|>") && !first_piece.contains("<|eot_id|>") {
            let processed = filter.process(&first_piece);
            if !processed.is_empty() {
                print!("{}", processed);
                io::stdout().flush()?;
            }
            generated_text.push_str(&first_piece);
        }
    }

    while token != model.token_eos() && n_cur < config.inference.n_ctx as i32 {
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch).context("Failed to decode token")?;
        token = sampler.sample(&ctx, 0);
        sampler.accept(token);

        if model.token_attr(token).contains(LlamaTokenAttr::Control) {
            break;
        }

        let piece = token_to_string(model, token);
        if piece.contains("<|im_end|>") || piece.contains("<|eot_id|>") {
            break;
        }
        let processed = filter.process(&piece);
        if !processed.is_empty() {
            print!("{}", processed);
            io::stdout().flush()?;
        }
        generated_text.push_str(&piece);
    }
    let remaining = filter.flush();
    if !remaining.is_empty() {
        print!("{}", remaining);
        io::stdout().flush()?;
    }
    println!();
    Ok(generated_text)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.output_only {
        llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    }
    let bin_dir = get_binary_dir().unwrap_or_else(|_| PathBuf::from("."));
    
    if !args.output_only {
        println!("Binary directory: {}", bin_dir.display());
    }

    let mut config = load_config(&bin_dir)?;

    if args.list_tiers {
        println!("Available model tiers:");
        let mut keys: Vec<&String> = config.tiers.keys().collect();
        let order = ["pico", "nano", "micro", "tiny", "mini", "small", "medium", "large", "xl", "xxl"];
        keys.sort_by_key(|key| {
            order.iter().position(|&x| x == key.as_str()).unwrap_or(usize::MAX)
        });
        for key in keys {
            if let Some(t) = config.tiers.get(key) {
                println!("  - {:<15} : {} (template: {})", key, t.name, t.template);
            }
        }
        return Ok(());
    }

    let tier_name = args.tier.clone().unwrap_or_else(|| config.default_tier.clone());
    let mut tier = config
        .tiers
        .get(&tier_name)
        .cloned()
        .context(format!("Tier '{}' not found in config.json", tier_name))?;

    // Apply CLI overrides if provided
    if let Some(temp) = args.temp {
        config.inference.temp = temp;
    }
    if let Some(top_p) = args.top_p {
        config.inference.top_p = top_p;
    }
    if let Some(ctx) = args.ctx {
        config.inference.n_ctx = ctx;
    }
    if let Some(gpu_layers) = args.gpu_layers {
        tier.n_gpu_layers = gpu_layers;
    }

    let model_path = bin_dir.join(&tier.filename);
    if !model_path.exists() {
        if !args.output_only {
            println!("Model file not found. Starting download...");
        }
        download_model(&tier.url, &model_path)?;
    }

    if !args.output_only {
        println!("Loading model: {}...", tier.name);
    }

    let mut backend = LlamaBackend::init().context("Failed to initialize backend")?;
    if args.output_only {
        backend.void_logs();
    }

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(tier.n_gpu_layers as u32)
        .with_use_mmap(config.inference.use_mmap)
        .with_use_mlock(config.inference.use_mlock);

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("Failed to load model from file")?;

    let default_system = "You are a helpful reasoning assistant.";

    if let Some(chat_path_str) = args.chat {
        let chat_path = Path::new(&chat_path_str);
        let mut history = load_chat_history(chat_path).unwrap_or_default();

        if let Some(ref prompt) = args.prompt {
            history.push(ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            });
            let formatted = format_chat_history(&tier.template, default_system, &history);
            let reply = run_inference(&model, &backend, &config, &tier, &formatted, args.show_thinking)?;
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: reply,
            });
            save_chat_history(chat_path, &history)?;
        } else {
            if !args.output_only {
                println!("Starting interactive chat mode with history file: {}. Type 'exit' to quit.", chat_path_str);
            }
            loop {
                print!("\n> ");
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                if input.is_empty() {
                    continue;
                }
                history.push(ChatMessage {
                    role: "user".to_string(),
                    content: input.to_string(),
                });
                let formatted = format_chat_history(&tier.template, default_system, &history);
                match run_inference(&model, &backend, &config, &tier, &formatted, args.show_thinking) {
                    Ok(reply) => {
                        history.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: reply,
                        });
                        let _ = save_chat_history(chat_path, &history);
                    }
                    Err(e) => {
                        eprintln!("Inference error: {:?}", e);
                    }
                }
            }
        }
    } else {
        if let Some(ref prompt) = args.prompt {
            let formatted = format_prompt(&tier.template, default_system, prompt);
            run_inference(&model, &backend, &config, &tier, &formatted, args.show_thinking)?;
        } else {
            println!("Starting interactive chat mode. Type 'exit' to quit.");
            loop {
                print!("\n> ");
                io::stdout().flush()?;
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                if input.is_empty() {
                    continue;
                }
                let formatted = format_prompt(&tier.template, default_system, input);
                if let Err(e) = run_inference(&model, &backend, &config, &tier, &formatted, args.show_thinking) {
                    eprintln!("Inference error: {:?}", e);
                }
            }
        }
    }

    Ok(())
}
