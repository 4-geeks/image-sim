# image-sim
search in database using deep image similarity

# Usage
At first we need to extract feature vector for each books in database. feature vectors will saved in a json format file. after that we can use this module to search in created database. 

### Create feature vector dictionary:
put all book images in `databse` folder.
run `index.py`

### Use in offline mode:
Put query images in `queries` folder and run `search.py` to see the results in `results` folder.

### use as an API:
run `server.py` and use modify `client.html` and run it in client side.
##
**Notice:** this task use pretrained VGG-16 and when you run each of this modules for first time, the VGG-16 weights will be downloaded in `.cache/torch/hub/checkpoints`.
