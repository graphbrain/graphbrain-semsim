sudo cp supervisor_config/run_script.conf /etc/supervisor/conf.d/run_script.conf

sudo supervisorctl reread
sudo supervisorctl update

tail -f logs/run_script.err.log