#run_simulate.py
import numpy as np
import pandas as pd

from data import TimeSeriesData, timedata
import gru
from simple_shap import *


def run(args):

	data_dir = ''

	df = pd.read_csv(args.data_dir / 'AIR.FR-d-20200701.csv', sep=' ', names=['name', 'date', 'time', 'val', 'extra'])
	df.drop('extra', axis=1, inplace=True)
	df['datetime'] = df[['date', 'time']].agg(' '.join, axis=1)
	df['datetime'] = df['datetime'].astype('datetime64[ns]')
	df.drop(['date', 'time'], axis=1, inplace=True)

	# change 'name' if needed
	ts = TimeSeriesData(df, name='BNP', h=args.h,y=args.y, k=args.k)
	ts.prepare_data()
	datadf = ts.df
	print('Data loaded ...')


	times_gru, shaps_gru = simulate_shap(args.nfeat, datadf.iloc[:5000,:-1],datadf.iloc[:5000,-1], model=args.model)

	print('--------------- mean|SHAP value| ---------------')
	for i, msv in enumerate(shaps_gru):
		print(f'\tfeat{i} = {abs(msv)}')

	print('---------------'*3)
	print(f"Average time: {np.mean(times_gru)}  seconds")

	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	featnames = [f'feat{i}' for i in range(args.nfeat)]
	ax.barh(featnames,list(map(abs,shaps_gru)), align='center')
	ax.set_ylabel('Features')
	ax.set_xlabel('Absolute SHAP')
	ax.set_title(f'SHAP_{args.model}_nfeat{args.nfeat}_k{args.k}')
	ax.invert_yaxis()
	plt.show(block=True)


if __name__ == '__main__':


	import argparse, pathlib

	parser = argparse.ArgumentParser(description="Running SHAP Simulations...")
	parser.add_argument('--data_dir', required=True, type=pathlib.Path, help='Location to dataset files')
	parser.add_argument('--h', type=int, default = 5, help='Number of input feature')
	parser.add_argument('--y', type=int,default=1, help='Output sequence length')
	parser.add_argument('--k', type=int, default=4, help='Jumps, k=1 means 15secs')
	parser.add_argument('--nfeat', type=int,required=True, help='No of Features to run Simulation')
	parser.add_argument('--model', required=True, help='Model used for Simulation')

	args = parser.parse_args()

	run(args)
