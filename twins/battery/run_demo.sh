#!/usr/bin/env bash
# One command: MQTT broker + twin service + web page + replay of one cell.
# Usage: ./run_demo.sh [test-cell index, default 27] [cycles per second, default 20]
set -e
cd "$(dirname "$0")"
CELL=${1:-27}; RATE=${2:-20}
trap 'kill $(jobs -p) 2>/dev/null' EXIT
amqtt > /dev/null 2>&1 &
sleep 2
python live/twin_service.py 0 &
sleep 2
streamlit run live/app.py --server.headless true --browser.gatherUsageStats false &
sleep 3
echo "Open http://localhost:8501"
python live/replayer.py --cell "$CELL" --rate "$RATE"
echo "Replay finished. Page stays up; press Ctrl+C to stop."
wait
