# Kitsu Database Sync Tool

Downloads and imports the Kitsu anime database dump for local use with Zantetsu.

## Overview

Kitsu provides weekly database dumps at:
- **URL**: `https://f002.backblazeb2.com/file/kitsu-dumps/latest.sql.gz`
- **Format**: PostgreSQL SQL dump (compressed with gzip)
- **Update Frequency**: Weekly

## Quick Start

### Prerequisites

- PostgreSQL 12+ installed and running
- `curl`, `gzip`, `psql` commands available
- Database user with CREATE DATABASE privileges

### Using the Shell Script

```bash
# Download the latest dump
./tools/kitsu-db-sync.sh download

# Import to PostgreSQL
./tools/kitsu-db-sync.sh import

# Or do both in one command
./tools/kitsu-db-sync.sh reset

# Check status
./tools/kitsu-db-sync.sh status

# With password (e.g., Docker PostgreSQL)
KITSU_DB_PASSWORD=root ./tools/kitsu-db-sync.sh reset
```

### Using the Rust CLI

```bash
# Build the tool
cargo build --release -p kitsu-sync

# Download
cargo run -p kitsu-sync -- download

# Import
cargo run -p kitsu-sync -- import

# Full reset
cargo run -p kitsu-sync -- reset
```

## Configuration

### Automatic Database Creation

The tool will **automatically create the target database** if it doesn't exist. You just need:
1. A running PostgreSQL server
2. A user with `CREATE DATABASE` privilege
3. Access to the `postgres` database (used to create the target database)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KITSU_DB_HOST` | `localhost` | PostgreSQL host |
| `KITSU_DB_PORT` | `5432` | PostgreSQL port |
| `KITSU_DB_NAME` | `kitsu_development` | Database name (auto-created if not exists) |
| `KITSU_DB_USER` | `postgres` | Database user |
| `KITSU_DB_PASSWORD` | - | Database password |
| `KITSU_DUMP_DIR` | `~/.local/share/zantetsu/kitsu-dumps` | Download directory |
| `PGPASSWORD` | - | Alternative password env var |

### Examples

```bash
# Use custom database
export KITSU_DB_NAME=my_anime_db
export KITSU_DB_USER=anime_user
./tools/kitsu-db-sync.sh reset

# With password
export KITSU_DB_PASSWORD=mypassword
./tools/kitsu-db-sync.sh import

# Using CLI with options
cargo run -p kitsu-sync -- -H mydb.example.com -p 5433 -U admin -d kitsu reset

# Docker PostgreSQL (password: root)
cargo run -p kitsu-sync -- -P root reset

# Auto-create a new database (no need to create manually!)
cargo run -p kitsu-sync -- -d my_new_anime_db -P mypassword reset
```

## Docker Setup

For local development with Docker:

```yaml
# docker-compose.yml
version: '3.8'
services:
  db:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_PASSWORD: root
      POSTGRES_DB: kitsu_development
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  adminer:
    image: adminer
    restart: always
    ports:
      - "8080:8080"

volumes:
  postgres_data:
```

Start the services:
```bash
docker-compose up -d
```

Then sync the database:
```bash
# The default password is 'root' as configured in docker-compose.yml
cargo run -p kitsu-sync -- -P root reset

# Or with the shell script
KITSU_DB_PASSWORD=root ./tools/kitsu-db-sync.sh reset
```

Access Adminer UI at http://localhost:8080 to browse the data:
- System: PostgreSQL
- Server: db
- Username: postgres
- Password: root
- Database: kitsu_development

## Database Schema

After import, the database contains tables including:
- `anime` - Anime titles and metadata
- `manga` - Manga titles and metadata
- `characters` - Character information
- `episodes` - Episode data
- `mappings` - External ID mappings (MAL, AniList, etc.)

## Troubleshooting

### Connection refused
```
psql: error: connection to server at "localhost" (::1), port 5432 failed
```
Make sure PostgreSQL is running:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

### Permission denied
```
ERROR:  permission denied to create database
```
Use a superuser or grant CREATE DATABASE privilege:
```sql
ALTER USER your_user CREATEDB;
```

### Download fails
Check your internet connection and try again. The dump is hosted on Backblaze B2.

## Data Usage

⚠️ **Important**: This dump is for development and personal use only. 
- Contains anime/manga metadata from Kitsu
- Updated weekly from production
- Do not redistribute the dump files
- Follow Kitsu's Terms of Service

## Integration with Zantetsu

The trainer crate (`zantetsu-trainer`) will use this data to:
1. Build title-to-ID mappings for semantic search
2. Create ground truth for model training
3. Generate vector embeddings for anime titles

See the main README for integration details.
