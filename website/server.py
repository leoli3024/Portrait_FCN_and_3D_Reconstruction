from flask import *
from os.path import join, dirname, realpath
import os


UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/images/uploads')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    print("got to main")
    return render_template('index.html')

@app.route('/post', methods=['GET', 'POST'])
def post():
    ### run ML shit here
    file = request.files['image_uploads']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    return render_template('post_result.html',
                            image=os.path.join('../static/images/uploads', file.filename),
                            trimap=os.path.join('../static/images/', "trimap.jpg"),
                            alpha=os.path.join('../static/images/', "alpha.jpg"),
                            alphabackground=os.path.join('../static/images/', "example.jpg"),
                            alphabackgroundtwo=os.path.join('../static/images/', "example2.jpg"))



if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, render_template, request
# from werkzeug import secure_filename
# app = Flask(__name__)

# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
    
# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       return 'file uploaded successfully'
        
# if __name__ == '__main__':
#    app.run(debug = True)