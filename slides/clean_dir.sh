#!/usr/bin/env bash
# Clean LaTeX auxiliary files and minted caches
# After run:
# xelatex -shell-escape name_file.tex

# List of extensions to remove
aux_exts=("aux" "log" "nav" "out" "snm" "toc" "vrb")

echo "🧹 Cleaning LaTeX build artifacts..."

# Remove LaTeX auxiliary files
for ext in "${aux_exts[@]}"; do
  rm -f -- *."$ext"
done

# Remove minted cache directories
rm -rf -- _minted _minted-* *minted

echo "✅ Cleanup complete."
