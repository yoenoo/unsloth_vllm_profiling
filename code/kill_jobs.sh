# nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
ps -au # look up running processes
ps auxf

# 1) Find a busy worker and get its process group id (PGID)
PGID=$(ps -o pgid= -p 2132370 | tr -d ' ')

# 2) Try graceful shutdown of the entire process group
kill -TERM -$PGID
sleep 5

# 3) If still alive, force kill the whole group
kill -KILL -$PGID