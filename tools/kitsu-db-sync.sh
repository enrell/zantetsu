#!/usr/bin/env bash
#
# Kitsu Database Dump Download and Import Script
# Downloads the latest Kitsu database dump and imports it to PostgreSQL
#

set -euo pipefail

# Configuration
DUMP_URL="https://f002.backblazeb2.com/file/kitsu-dumps/latest.sql.gz"
DUMP_DIR="${HOME}/.local/share/zantetsu/kitsu-dumps"
DUMP_FILE="${DUMP_DIR}/latest.sql.gz"
SQL_FILE="${DUMP_DIR}/latest.sql"

# Database configuration
DB_NAME="${KITSU_DB_NAME:-root}"
DB_USER="${KITSU_DB_USER:-root}"
DB_HOST="${KITSU_DB_HOST:-127.0.0.1}"
DB_PORT="${KITSU_DB_PORT:-5432}"
DB_PASSWORD="${KITSU_DB_PASSWORD:-${PGPASSWORD:-root}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create dump directory
mkdir -p "${DUMP_DIR}"

show_help() {
    cat << EOF
Kitsu Database Dump Manager

Usage: $0 [COMMAND]

Commands:
    download    Download the latest Kitsu database dump
    import      Import the dump to PostgreSQL (requires download first)
    reset       Download and import (full reset)
    clean       Remove downloaded dump files
    status      Check download status and database connection
    help        Show this help message

Environment Variables:
    KITSU_DB_NAME       Database name (default: kitsu_development)
    KITSU_DB_USER       Database user (default: postgres)
    KITSU_DB_HOST       Database host (default: localhost)
    KITSU_DB_PORT       Database port (default: 5432)
    KITSU_DB_PASSWORD   PostgreSQL password (default: empty)
    PGPASSWORD          Alternative for password (backward compatible)

Examples:
    $0 download
    $0 import
    $0 reset
    
    # With Docker PostgreSQL
    KITSU_DB_PASSWORD=root $0 reset
    
    # Custom host with password
    KITSU_DB_HOST=myhost KITSU_DB_PASSWORD=mypass $0 import
EOF
}

check_dependencies() {
    local missing=()
    
    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    fi
    
    if ! command -v psql &> /dev/null; then
        missing+=("postgresql-client")
    fi
    
    if ! command -v gunzip &> /dev/null; then
        missing+=("gzip")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing[*]}"
        log_info "Install with: sudo apt-get install ${missing[*]}"
        exit 1
    fi
}

# Helper function to run psql with password
run_psql() {
    local db="${1:-postgres}"
    shift || true
    
    # Export password for psql (works for both local and Docker PostgreSQL)
    export PGPASSWORD="${DB_PASSWORD}"
    
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${db}" "$@"
    local exit_code=$?
    
    unset PGPASSWORD
    return $exit_code
}

check_database_connection() {
    log_info "Checking database connection to ${DB_HOST}:${DB_PORT}..."
    
    if ! run_psql "postgres" -c "SELECT 1;" > /dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL at ${DB_HOST}:${DB_PORT}"
        log_info "Make sure PostgreSQL is running and credentials are correct"
        log_info "Set KITSU_DB_PASSWORD environment variable if password is required"
        return 1
    fi
    
    log_info "Database connection successful"
    return 0
}

download_dump() {
    log_info "Downloading Kitsu database dump..."
    log_info "URL: ${DUMP_URL}"
    log_info "Target: ${DUMP_FILE}"
    
    # Check if file already exists
    if [ -f "${DUMP_FILE}" ]; then
        log_warn "Dump file already exists: ${DUMP_FILE}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Download cancelled"
            return 0
        fi
        rm -f "${DUMP_FILE}"
    fi
    
    # Download with progress
    if ! curl -L --progress-bar -o "${DUMP_FILE}.tmp" "${DUMP_URL}"; then
        log_error "Download failed"
        rm -f "${DUMP_FILE}.tmp"
        return 1
    fi
    
    mv "${DUMP_FILE}.tmp" "${DUMP_FILE}"
    
    # Show file size
    local size=$(du -h "${DUMP_FILE}" | cut -f1)
    log_info "Download complete: ${size}"
    
    # Verify it's a valid gzip file
    if ! gunzip -t "${DUMP_FILE}" 2>/dev/null; then
        log_error "Downloaded file is not a valid gzip archive"
        rm -f "${DUMP_FILE}"
        return 1
    fi
    
    log_info "Dump file verified successfully"
}

extract_dump() {
    log_info "Extracting dump file..."
    
    if [ ! -f "${DUMP_FILE}" ]; then
        log_error "Dump file not found: ${DUMP_FILE}"
        log_info "Run: $0 download"
        return 1
    fi
    
    # Remove old SQL file if exists
    rm -f "${SQL_FILE}"
    
    # Extract
    if ! gunzip -c "${DUMP_FILE}" > "${SQL_FILE}"; then
        log_error "Extraction failed"
        rm -f "${SQL_FILE}"
        return 1
    fi
    
    local size=$(du -h "${SQL_FILE}" | cut -f1)
    log_info "Extraction complete: ${size}"
}

import_dump() {
    log_info "Importing Kitsu database dump..."
    log_info "Database: ${DB_NAME}"
    log_info "Host: ${DB_HOST}:${DB_PORT}"
    
    if [ ! -f "${SQL_FILE}" ]; then
        log_warn "SQL file not found, extracting from dump..."
        if ! extract_dump; then
            return 1
        fi
    fi
    
    # Check database connection
    if ! check_database_connection; then
        return 1
    fi
    
    # Check if target database exists
    log_info "Checking if database '${DB_NAME}' exists..."
    if ! run_psql "postgres" -c "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}';" | grep -q "1"; then
        log_info "Database '${DB_NAME}' does not exist. Creating it..."
        if ! run_psql "postgres" -c "CREATE DATABASE ${DB_NAME};"; then
            log_error "Failed to create database '${DB_NAME}'"
            log_info "Make sure user '${DB_USER}' has CREATE DATABASE privilege"
            return 1
        fi
        log_info "Database '${DB_NAME}' created successfully"
    else
        log_info "Database '${DB_NAME}' already exists"
    fi
    
    # Import the dump
    log_info "Importing SQL dump to '${DB_NAME}' (this may take several minutes)..."
    if ! run_psql "${DB_NAME}" -f "${SQL_FILE}"; then
        log_error "Import failed"
        return 1
    fi
    
    log_info "Import complete!"
    
    # Show table counts
    log_info "Database statistics:"
    run_psql "${DB_NAME}" -c "
        SELECT 
            'anime' as table_name, COUNT(*) as count FROM anime
        UNION ALL
        SELECT 'manga', COUNT(*) FROM manga
        UNION ALL
        SELECT 'characters', COUNT(*) FROM characters
        UNION ALL
        SELECT 'episodes', COUNT(*) FROM episodes
        ORDER BY table_name;
    " 2>/dev/null || log_warn "Could not retrieve statistics"
}

clean_dumps() {
    log_info "Cleaning up downloaded dump files..."
    
    local removed=0
    
    if [ -f "${DUMP_FILE}" ]; then
        rm -f "${DUMP_FILE}"
        log_info "Removed: ${DUMP_FILE}"
        removed=$((removed + 1))
    fi
    
    if [ -f "${SQL_FILE}" ]; then
        rm -f "${SQL_FILE}"
        log_info "Removed: ${SQL_FILE}"
        removed=$((removed + 1))
    fi
    
    if [ $removed -eq 0 ]; then
        log_info "No dump files to clean"
    else
        log_info "Cleaned ${removed} file(s)"
    fi
}

show_status() {
    log_info "Kitsu Database Dump Status"
    echo "=========================="
    
    # Check dump file
    if [ -f "${DUMP_FILE}" ]; then
        local size=$(du -h "${DUMP_FILE}" | cut -f1)
        local date=$(stat -c %y "${DUMP_FILE}" 2>/dev/null || stat -f %Sm "${DUMP_FILE}" 2>/dev/null)
        echo "Dump file: ${DUMP_FILE}"
        echo "  Size: ${size}"
        echo "  Downloaded: ${date}"
    else
        echo "Dump file: Not downloaded"
    fi
    
    if [ -f "${SQL_FILE}" ]; then
        local size=$(du -h "${SQL_FILE}" | cut -f1)
        echo "SQL file: ${SQL_FILE}"
        echo "  Size: ${size}"
    else
        echo "SQL file: Not extracted"
    fi
    
    echo ""
    echo "Database:"
    echo "  Name: ${DB_NAME}"
    echo "  Host: ${DB_HOST}:${DB_PORT}"
    echo "  User: ${DB_USER}"
    
    if check_database_connection > /dev/null 2>&1; then
        echo "  Status: Connected"
        
        # Check if database exists
        if run_psql "${DB_NAME}" -c "SELECT 1;" > /dev/null 2>&1; then
            echo "  Database: Exists"
            
            # Get table counts
            local anime_count=$(run_psql "${DB_NAME}" -t -c "SELECT COUNT(*) FROM anime;" 2>/dev/null | tr -d ' ')
            if [ -n "${anime_count}" ]; then
                echo "  Anime entries: ${anime_count}"
            fi
        else
            echo "  Database: Not created"
        fi
    else
        echo "  Status: Cannot connect"
    fi
}

# Main command handler
case "${1:-help}" in
    download)
        check_dependencies
        download_dump
        ;;
    import)
        check_dependencies
        import_dump
        ;;
    reset)
        check_dependencies
        download_dump && extract_dump && import_dump
        ;;
    clean)
        clean_dumps
        ;;
    status)
        check_dependencies
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
