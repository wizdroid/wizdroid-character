#!/bin/bash
# Quick commit script for wizdroid-character
# Usage: ./quick-commit.sh "your commit message"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"your commit message\""
    echo "Example: $0 \"fix: resolved character prompt issue\""
    exit 1
fi

MESSAGE="$1"

echo "Staging all modified files..."
git add -A

echo "Committing with message: $MESSAGE"
git commit -m "$MESSAGE"

echo "Pushing to origin/main..."
git push origin main

echo "Done!"
