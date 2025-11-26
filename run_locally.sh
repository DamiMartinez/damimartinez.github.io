#!/bin/bash
echo "Starting Jekyll site using Docker..."
docker run --rm \
  --volume="$PWD:/srv/jekyll" \
  -p 4000:4000 \
  jekyll/builder:latest \
  jekyll serve --watch --force_polling
