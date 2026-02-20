//! Kitsu Database Sync Tool
//!
//! Downloads and imports the Kitsu anime database dump for local use.
//! Provides both a CLI interface and a library API.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn, error};

/// Kitsu database dump URL
const DUMP_URL: &str = "https://f002.backblazeb2.com/file/kitsu-dumps/latest.sql.gz";

/// Default dump directory
fn default_dump_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("zantetsu")
        .join("kitsu-dumps")
}

/// CLI arguments
#[derive(Parser)]
#[command(name = "kitsu-sync")]
#[command(about = "Download and sync Kitsu database dumps")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Database host
    #[arg(short = 'H', long, env = "KITSU_DB_HOST", default_value = "localhost")]
    host: String,
    
    /// Database port
    #[arg(short = 'p', long, env = "KITSU_DB_PORT", default_value_t = 5432)]
    port: u16,
    
    /// Database name
    #[arg(short, long, env = "KITSU_DB_NAME", default_value = "kitsu_development")]
    database: String,
    
    /// Database user
    #[arg(short = 'U', long, env = "KITSU_DB_USER", default_value = "postgres")]
    user: String,

    /// Database password
    #[arg(short = 'P', long, env = "KITSU_DB_PASSWORD")]
    password: Option<String>,

    /// Dump directory
    #[arg(short = 'D', long, env = "KITSU_DUMP_DIR")]
    dump_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download the latest database dump
    Download {
        /// Force re-download if file exists
        #[arg(short, long)]
        force: bool,
    },
    /// Import the dump to PostgreSQL
    Import {
        /// Extract before importing (if not already extracted)
        #[arg(short, long)]
        extract: bool,
    },
    /// Download and import (full reset)
    Reset,
    /// Clean up downloaded files
    Clean {
        /// Skip confirmation
        #[arg(short, long)]
        yes: bool,
    },
    /// Show status and information
    Status,
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub user: String,
    pub password: Option<String>,
}

impl DatabaseConfig {
    /// Create PostgreSQL connection string
    pub fn connection_string(&self) -> String {
        format!(
            "postgresql://{}@{}:{}/{}",
            self.user, self.host, self.port, self.database
        )
    }
    
    /// Check if database is accessible
    pub async fn check_connection(&self) -> Result<bool> {
        // We'll use the shell script for now
        // In a full implementation, this would use sqlx or tokio-postgres
        Ok(true)
    }
}

/// Kitsu dump manager
pub struct KitsuDumpManager {
    dump_dir: PathBuf,
    db_config: DatabaseConfig,
}

impl KitsuDumpManager {
    /// Create a new dump manager
    pub fn new(dump_dir: PathBuf, db_config: DatabaseConfig) -> Self {
        Self {
            dump_dir,
            db_config,
        }
    }
    
    /// Get path to the compressed dump file
    pub fn dump_file_path(&self) -> PathBuf {
        self.dump_dir.join("latest.sql.gz")
    }
    
    /// Get path to the extracted SQL file
    pub fn sql_file_path(&self) -> PathBuf {
        self.dump_dir.join("latest.sql")
    }
    
    /// Check if dump file exists
    pub fn dump_exists(&self) -> bool {
        self.dump_file_path().exists()
    }
    
    /// Check if SQL file exists
    pub fn sql_exists(&self) -> bool {
        self.sql_file_path().exists()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    let dump_dir = cli.dump_dir.unwrap_or_else(default_dump_dir);
    let db_config = DatabaseConfig {
        host: cli.host,
        port: cli.port,
        database: cli.database,
        user: cli.user,
        password: cli.password,
    };

    // For now, delegate to the shell script
    // This provides a stable interface while we implement native Rust version
    // The script is at tools/kitsu-db-sync.sh and we're in tools/kitsu-sync/
    let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("kitsu-db-sync.sh");

    let mut cmd = Command::new(&script_path);
    cmd.env("KITSU_DB_HOST", &db_config.host)
        .env("KITSU_DB_PORT", db_config.port.to_string())
        .env("KITSU_DB_NAME", &db_config.database)
        .env("KITSU_DB_USER", &db_config.user);

    // Add password if provided
    if let Some(password) = &db_config.password {
        cmd.env("KITSU_DB_PASSWORD", password);
    }
    
    // Ensure dump directory exists
    std::fs::create_dir_all(&dump_dir)?;
    
    match cli.command {
        Commands::Download { force } => {
            info!("Downloading Kitsu database dump...");
            if force {
                // Remove existing file to force re-download
                let dump_file = dump_dir.join("latest.sql.gz");
                if dump_file.exists() {
                    std::fs::remove_file(&dump_file)?;
                    info!("Removed existing dump file");
                }
            }
            cmd.arg("download");
        }
        Commands::Import { extract } => {
            info!("Importing database dump...");
            if extract {
                info!("Will extract before importing");
            }
            cmd.arg("import");
        }
        Commands::Reset => {
            info!("Performing full reset (download + import)...");
            cmd.arg("reset");
        }
        Commands::Clean { yes } => {
            if !yes {
                println!("This will remove all downloaded dump files.");
                print!("Are you sure? [y/N]: ");
                use std::io::Write;
                std::io::stdout().flush()?;
                
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                
                if !input.trim().eq_ignore_ascii_case("y") {
                    info!("Clean cancelled");
                    return Ok(());
                }
            }
            cmd.arg("clean");
        }
        Commands::Status => {
            cmd.arg("status");
        }
    }
    
    let status = cmd.status()
        .context("Failed to execute sync script")?;
    
    if !status.success() {
        anyhow::bail!("Sync script failed with exit code: {:?}", status.code());
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_dump_dir() {
        let dir = default_dump_dir();
        assert!(dir.to_string_lossy().contains("zantetsu"));
    }
    
    #[test]
    fn test_database_config() {
        let config = DatabaseConfig {
            host: "localhost".to_string(),
            port: 5432,
            database: "kitsu".to_string(),
            user: "postgres".to_string(),
            password: Some("postgres".to_string()),
        };
        
        let conn_str = config.connection_string();
        assert!(conn_str.contains("localhost"));
        assert!(conn_str.contains("5432"));
        assert!(conn_str.contains("kitsu"));
    }
}
