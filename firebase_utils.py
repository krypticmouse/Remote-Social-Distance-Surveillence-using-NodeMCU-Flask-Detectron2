from firebase import firebase

firebase = firebase.FirebaseApplication('https://socialdistancing-9b13e.firebaseio.com/', None)

def get_data(id):
    return list(firebase.get(str(id),'').values())[-1]

def post_data(id, is_violation):
    firebase.put('/',str(id), is_violation)

def flush_database(id):
    firebase.delete(str(id), '')