#!/bin/bash

source .env

curl -X PUT \
-H "Authorization: Bearer ${CHANNEL_ACCESS_TOKEN}" \
-H 'Content-Type:application/json' \
-d '{"endpoint":"'"${LINE_BOT_WEBHOOK_URL}/callback"'"}' \
https://api.line.me/v2/bot/channel/webhook/endpoint