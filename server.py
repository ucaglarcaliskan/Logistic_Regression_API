#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:49:18 2019

@author: caglar
"""
import asyncio
from flask import Flask,jsonify
from flask_restful import Resource, Api, reqparse
import werkzeug, os
import logistic as lg

loop = asyncio.get_event_loop()
app = Flask(__name__)
api = Api(app)

datasets = []
UPLOAD_FOLDER = 'datasets'
parser = reqparse.RequestParser()
#parser.add_argument('name')
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')



class PhotoUpload(Resource):
    decorators=[]

    def post(self,filename):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        file = data['file']
        #name = data['name']
        if file:
            #filename = filename
            file.save(os.path.join(UPLOAD_FOLDER,filename))
            #Model()
            return {
                    'data':'',
                    'message':'Dataset uploaded',
                    'status':'success'
                    }
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }

class GetColumns(Resource):
    def get(self,dataset):
        columns = lg.load_dataset(dataset)
        return {'columns':columns}
    
class SetLabel(Resource):
    def get(self,label):
        lg.init_dataset(label)
        return 'Label is set'
        
class Train(Resource):
    def get(self):
        fpr, tpr =lg.gradient_descent_g()
        #lg.roc_g()
        return  {'fpr': fpr,
                 'tpr': tpr}
            
class ListDataset(Resource):
    def get(self):
        datasets = []
        for root, dirs, files in os.walk("./datasets"):
            for filename in files:
                datasets.append(filename)
        return {'datasets': datasets }
                
        

api.add_resource(PhotoUpload,'/uploader/<string:filename>')

api.add_resource(Train,'/train')

api.add_resource(ListDataset, '/list')
api.add_resource(GetColumns, '/dataset/<string:dataset>')
api.add_resource(SetLabel, '/label/<label>')

if __name__ == '__main__':
    app.run(debug=True)