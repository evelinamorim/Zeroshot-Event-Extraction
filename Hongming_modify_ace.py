import ujson as json

folder_name = '/shared/lyuqing/probing_for_event/data/ACE_oneie/en/event_only'

with open(folder_name+'/'+'dev.event.json', 'r') as f:
    tmp_data = json.load(f)

for tmp_example in tmp_data:
    print(tmp_example)
    break
