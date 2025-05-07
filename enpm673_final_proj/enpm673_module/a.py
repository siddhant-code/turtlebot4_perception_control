import configparser
config = configparser.ConfigParser()
config.read('enpm673_final_proj/enpm673_module/config.ini')
print(config['DEFAULT']['kl'])