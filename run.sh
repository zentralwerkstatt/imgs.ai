export IMGS_CONFIG=${2:-config}
LRU_CACHE_CAPACITY=1 gunicorn --timeout 600 -b 0.0.0.0:${1:-5000} --access-logfile logs/access_`date +"%Y-%m-%d"`.log app:app