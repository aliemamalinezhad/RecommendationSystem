import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating = 4.0)


print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss = 'wrap')

model.fit(data['train'],epochs = 30, num_threads = 2)

def sample_recommendation(model,data,user_ids):
	
	n_users,n_items = data['train'].shape

	for user_id in user_ids:

 		known_positive = data['item_labels'][data["train"].tocsr()[user_id].indices]

 		scores = model.predict(user_id,np.arange(n_items))

 		top_items = data['item_labels'][np.argsort(-scores)]

 		print('User %s' % user_id)

 		print("   known positive")

 		for x in known_positive[:3]:
 			print("   %s" % x)

 		print("   Recommended:")

 		for x in top_items[:3]:
 			print("    %s" %x )

