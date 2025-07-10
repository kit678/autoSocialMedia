#!/bin/bash
export NVM_HOME="C:/ProgramData/nvm"
export NVM_SYMLINK="C:/ProgramData/nodejs"
export PATH="$PATH:/c/Users/hweizen/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1.1-full_build/bin/ffmpeg"
# Add this line here
#export NODE_FUNCTION_ALLOW_EXTERNAL=google-trends-api
export NODE_FUNCTION_ALLOW_EXTERNAL='@mozilla/readability,jsdom'
echo "DEBUG: Variable is set to -> $NODE_FUNCTION_ALLOW_EXTERNAL"

source "$NVM_HOME/nvm.sh" # Source NVM to make it available
nvm use 22.15.0
npx n8n start