import json
from google.cloud import datastore

class DataStorePipeline(object):

    def open_spider(self, spider):
        return

    def close_spider(self, spider):
        return


    def process_item(self, item, spider):
        # Create, populate and persist an entity
        # Instantiates a client
        datastore_client = datastore.Client()

        # The kind for the new entity
        kind = 'Article'
        # The name/ID for the new entity
        name = item['full_url']
        # The Cloud Datastore key for the new entity
        task_key = datastore_client.key(kind, name)

        # Prepares the new entity
        task = datastore.Entity(key=task_key)
        task.__dict__.update(item.__dict__)

        # Saves the entity
        datastore_client.put(task)

        print('Saved {}: {}'.format(task.key.name, task['title']))
        return item