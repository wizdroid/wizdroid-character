#!/bin/bash

# 🧙 Wizdroid Character - GitHub Checkin Script
# Usage: ./scripts/checkin.sh "Your commit message here"

if [ -z "$1" ]; then
    echo "❌ Error: No commit message provided"
    echo "Usage: ./scripts/checkin.sh \"Your commit message\""
    exit 1
fi

COMMIT_MESSAGE="$1"

echo "📊 Current Status:"
git status

echo ""
echo "📝 Committing with message: $COMMIT_MESSAGE"
echo ""

# Stage all changes
git add -A

# Commit with provided message
if git commit -m "$COMMIT_MESSAGE"; then
    echo ""
    echo "✅ Commit successful"
    echo ""
    echo "🚀 Pushing to GitHub..."
    
    # Push to remote
    if git push; then
        echo ""
        echo "✅ Push successful!"
        echo "🎉 Checkin complete!"
    else
        echo ""
        echo "❌ Push failed!"
        exit 1
    fi
else
    echo ""
    echo "❌ Commit failed (nothing to commit?)"
    git status
    exit 1
fi
