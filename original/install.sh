#!/bin/bash
# Install Rust toolchain for MLSys Challenge 2026

echo "Installing Rust toolchain..."

# Install rustup if not present
if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Update to latest stable
rustup update stable

# Verify installation
echo ""
echo "Rust installation complete:"
rustc --version
cargo --version

echo ""
echo "Building mlsys..."
cargo build --release

echo ""
echo "Running tests..."
cargo test

echo ""
echo "Done! Binary available at: target/release/mlsys"

