import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

data = fetch_movielens(min_rating = 4.0)


print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss = 'warp')

model.fit(data['train'],epochs = 30, num_threads = 2)
print("Train precision: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data['test'], k=5).mean())



def sample_recommendation(model,data,user_ids):
	
	n_users,n_items = data['train'].shape

	for user_id in user_ids:

 		known_positive = data['item_labels'][data["train"].tocsr()[user_id].indices]

 		scores = model.predict(user_id,np.arange(n_items))

 		top_items = data['item_labels'][np.argsort(-scores)]

 		print('User %s' % user_id)

 		print("   known positive")

 		for x in known_positive[:3]:
 			print("  %s" % x)

 		print("   Recommended:")

 		for x in top_items[:3]:
 			print("    %s"  % x)

sample_recommendation(model,data,[3,25,450])
