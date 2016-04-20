
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<pthread.h>
#include<ctime>
#include<sys/time.h>
#include<assert.h>
using namespace std;

//Ruobing Xie
//Representation Learning of Knowledge Graphs with Hierarchical Types
//Weighted Hierarchy Encoder + Soft Type Constraint

#define pi 3.1415926535897932384626433832795
#define THREADS_NUM 8

bool L1_flag=1;

int nepoch = 1000;		//iteration times
string transE_version = "unif";		//unif/bern
double type_weight = 0.9;		//proportional-declined weighting belta

//random
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)		//vector length
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

double norm(vector<double> &a)
{
	double x = vec_len(a);
	if (x>1)
	for (int ii=0; ii<a.size(); ii++)
			a[ii]/=x;
	return 0;
}

//parameters
string version;
char buf[100000],buf1[100000];
int relation_num,entity_num,type_num,domain_num;		//number
map<string,int> relation2id,entity2id,type2id,domain2id;
map<int,string> id2entity,id2relation,id2type,id2domain;
//relation-specific information: indicate the corresponding type head/tail should be in specific relation
vector<int> head_type_vec,tail_type_vec;
vector<int> head_domain_vec,tail_domain_vec;
int nbatches, batchsize;

map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;

int n,method;		//n：dimension of entity/relation
double res_triple,res_normal;		//loss function value
double res_thread_triple[THREADS_NUM], res_thread_normal[THREADS_NUM];		//loss for each thread
double count,count1;
double rate,rate_end,rate_begin;		//rate: dynamic learning rate
double margin;		//margin
double belta;
vector<int> fb_h,fb_l,fb_r;
vector<vector<double> > relation_vec,entity_vec;		//embeddings
vector<vector<double> > relation_tmp,entity_tmp;

vector<vector<int> > type_entity_list;		//record all entities a certain type has, for Soft type constraint
vector<int> type_entity_num;

//Freebase general hierarchical type structure: domain/type/topic
vector<vector<vector<double> > > domain_mat;		//sub-type matrices
vector<vector<vector<double> > > type_mat;
vector<vector<vector<double> > > domain_mat_tmp;
vector<vector<vector<double> > > type_mat_tmp;

vector<double> posErrorVec[THREADS_NUM];
vector<double> negErrorVec[THREADS_NUM];
vector<double> head_final_vec[THREADS_NUM];
vector<double> tail_final_vec[THREADS_NUM];

pthread_mutex_t mut_mutex;

void sgd();
void train_triple_mul(int, int, int, int, int, int, int);

vector<double> norm_tmp_vec[THREADS_NUM];		//norm_2中的变量

double norm_2(vector<double> &a, vector<vector<double> > &D, vector<vector<double> > &T, int tid)		//normalization
{
	double x=0;
	double der_1 = 0;
	//calc
	for(int i=0; i<n; i++)
	{
		double type_score = 0, domain_score = 0;
		double tmp = 0;
		for(int ii=0; ii<n; ii++)
		{
			type_score += T[i][ii] * a[ii];
		}
		for(int ii=0; ii<n; ii++)
		{
			domain_score += D[i][ii] * a[ii];
		}
		tmp = type_score * type_weight + domain_score * (1-type_weight);
		norm_tmp_vec[tid][i] = 2*tmp;
		x += sqr(tmp);
	}
	
	//gradient
	if(x>1)
	{
		double lamda=1;
		for(int i=0; i<n; i++)
		{
			der_1 = -rate*lamda*norm_tmp_vec[tid][i]*(1-type_weight);
			for(int ii=0; ii<n; ii++)
			{
				D[i][ii] += der_1 * a[ii];
				a[ii] += der_1 * D[i][ii];
			}
			der_1 = -rate*lamda*norm_tmp_vec[tid][i]*type_weight;
			for(int ii=0; ii<n; ii++)
			{
				T[i][ii] += der_1 * a[ii];
				a[ii] += der_1 * T[i][ii];
			}
		}
	}
}

int rand_max(int x)
{
	int res = (rand()*rand())%x;
	while (res<0)
		res+=x;
	return res;
}

map<pair<int,int>, map<int,int> > ok;

void add(int x,int y,int z)
{
	fb_h.push_back(x);
	fb_r.push_back(z);
	fb_l.push_back(y);
	ok[make_pair(x,z)][y]=1;		//positive mark
}

void run(int n_in,double rate_in,double margin_in,int method_in)
{
	//init
	n = n_in;
	rate = rate_in;
	margin = margin_in;
	method = method_in;
	
	relation_vec.resize(relation_num);
	for (int i=0; i<relation_vec.size(); i++)
		relation_vec[i].resize(n);
	entity_vec.resize(entity_num);
	for (int i=0; i<entity_vec.size(); i++)
		entity_vec[i].resize(n);
	relation_tmp.resize(relation_num);
	for (int i=0; i<relation_tmp.size(); i++)
		relation_tmp[i].resize(n);
	entity_tmp.resize(entity_num);
	for (int i=0; i<entity_tmp.size(); i++)
		entity_tmp[i].resize(n);
	
	//init by pre-trained entity/relation embeddings of TransE
	FILE* f1 = fopen(("../transE_res/entity2vec."+transE_version).c_str(),"r");
	for (int i=0; i<entity_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			fscanf(f1,"%lf",&entity_vec[i][ii]);
		norm(entity_vec[i]);
	}
	fclose(f1);

	FILE* f2 = fopen(("../transE_res/relation2vec."+transE_version).c_str(),"r");
	for (int i=0; i<relation_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			fscanf(f2,"%lf",&relation_vec[i][ii]);
	}
	fclose(f2);
	
	//init domain matrix
	domain_mat.resize(domain_num);
	domain_mat_tmp.resize(domain_num);
	for(int i=0; i<domain_num; i++)
	{
		domain_mat[i].resize(n);
		domain_mat_tmp[i].resize(n);
		for(int ii=0; ii<n; ii++)
		{
			domain_mat[i][ii].resize(n);
			domain_mat_tmp[i][ii].resize(n);
			for(int iii=0; iii<n; iii++)
			{
				if(ii==iii)
					domain_mat[i][ii][iii] = 1;
				else
					domain_mat[i][ii][iii] = 0;
			}
		}
	}
	//init type matrix
	type_mat.resize(type_num);
	type_mat_tmp.resize(type_num);
	for(int i=0; i<type_num; i++)
	{
		type_mat[i].resize(n);
		type_mat_tmp[i].resize(n);
		for(int ii=0; ii<n; ii++)
		{
			type_mat[i][ii].resize(n);
			type_mat_tmp[i][ii].resize(n);
			for(int iii=0; iii<n; iii++)
			{
				if(ii==iii)
					type_mat[i][ii][iii] = 1;
				else
					type_mat[i][ii][iii] = 0;
			}
		}
	}
	//init mid vec
	for(int i=0; i<THREADS_NUM; i++)
	{
		posErrorVec[i].resize(n);
		negErrorVec[i].resize(n);
		head_final_vec[i].resize(n);
		tail_final_vec[i].resize(n);
		norm_tmp_vec[i].resize(n);
	}
	
	
	mut_mutex = PTHREAD_MUTEX_INITIALIZER;
	sgd();
}

void *rand_sel(void *tid_void)		//multi-thread train
{
	long tid = (long) tid_void;
	for (int k=0; k<batchsize; k++)
	{
		int i=rand_max(fb_h.size());		//positive mark
		int j=rand_max(entity_num);		//negative entity
		double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
		if (method ==0)
			pr = 500;
		
		//negative sampling
		int flag_num = rand_max(1000);
		int temp_head_type = head_type_vec[fb_r[i]];
		int temp_tail_type = tail_type_vec[fb_r[i]];
		if (flag_num<pr)
		{
			if(rand_max(entity_num+10*type_entity_num[temp_tail_type]) > entity_num)		//Soft type constraint, parameter could be changed
			{
				int jj = rand_max(type_entity_num[temp_tail_type]);
				j = type_entity_list[temp_tail_type][jj];
			}
			while (ok.count(make_pair(fb_h[i],fb_r[i]))>0&&ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
				j=rand_max(entity_num);
			train_triple_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		}
		else
		{
			if(rand_max(entity_num+10*type_entity_num[temp_head_type]) > entity_num)		//Soft type constraint, parameter could be changed
			{
				int jj = rand_max(type_entity_num[temp_head_type]);
				j = type_entity_list[temp_head_type][jj];
			}
			while (ok.count(make_pair(j,fb_r[i]))>0&&ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
				j=rand_max(entity_num);
			train_triple_mul(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		}
		
		int rel_neg = rand_max(relation_num);		//negative relation
		while (ok.count(make_pair(fb_h[i], rel_neg))>0&& ok[make_pair(fb_h[i], rel_neg)].count(fb_l[i]) > 0)
			rel_neg = rand_max(relation_num);
		train_triple_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],fb_l[i],rel_neg,tid);
		
		//normalization
		norm(relation_tmp[fb_r[i]]);
		norm(relation_tmp[rel_neg]);
		norm(entity_tmp[fb_h[i]]);
		norm(entity_tmp[fb_l[i]]);
		norm(entity_tmp[j]);
		
		norm_2(entity_tmp[fb_h[i]], domain_mat_tmp[head_domain_vec[fb_r[i]]], type_mat_tmp[head_type_vec[fb_r[i]]], tid);
		norm_2(entity_tmp[fb_l[i]], domain_mat_tmp[tail_domain_vec[fb_r[i]]], type_mat_tmp[tail_type_vec[fb_r[i]]], tid);
		if(flag_num<pr)
			norm_2(entity_tmp[j], domain_mat_tmp[tail_domain_vec[fb_r[i]]], type_mat_tmp[tail_type_vec[fb_r[i]]], tid);
		else
			norm_2(entity_tmp[j], domain_mat_tmp[head_domain_vec[fb_r[i]]], type_mat_tmp[head_type_vec[fb_r[i]]], tid);
		norm_2(entity_tmp[fb_h[i]], domain_mat_tmp[head_domain_vec[rel_neg]], type_mat_tmp[head_type_vec[rel_neg]], tid);
		norm_2(entity_tmp[fb_l[i]], domain_mat_tmp[tail_domain_vec[rel_neg]], type_mat_tmp[tail_type_vec[rel_neg]], tid);
		
	}
}

void update_multithread()		//update loss
{
	//update
	for(int k = 0;k<THREADS_NUM;k++)
	{
		res_triple += res_thread_triple[k];
		res_normal += res_thread_normal[k];
	}
}

void sgd()		//mini-batch SGD
{
	res_triple=0;
	res_normal = 0;
	nbatches=100;		//block number
	batchsize = fb_h.size()/nbatches/THREADS_NUM;		//mini_batch size for each thread
	cout << "batchsize : " << batchsize << endl;
	double step = (rate_begin - rate_end) / (double)nepoch;		//dynamic learning rate setting
	cout << step << ' ' << nepoch << endl;
	rate = rate_begin;
	for (int epoch=0; epoch<nepoch; epoch++)
	{
		rate -= step;
		res_triple=0;
		res_normal = 0;
		for (int batch = 0; batch<nbatches; batch++)
		{
			for(int k = 0;k<THREADS_NUM;k++)		//init
			{
				res_thread_triple[k] = 0;
				res_thread_normal[k] = 0;
			}
			relation_tmp = relation_vec;
			entity_tmp = entity_vec;
			domain_mat_tmp = domain_mat;
			type_mat_tmp = type_mat;
			//multi-thread for train
			pthread_t threads[THREADS_NUM];
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_create(&threads[k], NULL, rand_sel, (void *)k);		//train
			}
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_join(threads[k], NULL);
			}
			//multi-thread for update
			relation_vec = relation_tmp;		//update
			entity_vec = entity_tmp;
			domain_mat = domain_mat_tmp;
			type_mat = type_mat_tmp;
			update_multithread();
			//cout << "update once : " << batch << endl;
		}
		//output
		cout<<"epoch:"<<epoch<<' '<<res_triple<< ' ' << res_normal << endl;
		FILE* f2 = fopen(("../res/relation2vec."+version).c_str(),"w");
		FILE* f3 = fopen(("../res/entity2vec."+version).c_str(),"w");
		FILE* f5 = fopen(("../res/typeMatrix."+version).c_str(),"w");
		FILE* f6 = fopen(("../res/domainMatrix."+version).c_str(),"w");
		for (int i=0; i<relation_num; i++)		//relation2vec
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
			fprintf(f2,"\n");
		}
		for (int i=0; i<entity_num; i++)		//entity_vec
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
			fprintf(f3,"\n");
		}
		for (int i=0; i<type_num; i++)		//sub-type matrix
		{
			for (int ii=0; ii<n; ii++)
			{
				for(int iii=0; iii<n; iii++)
					fprintf(f5,"%.6lf\t",type_mat[i][ii][iii]);
			}
			fprintf(f5,"\n");
		}
		for (int i=0; i<domain_num; i++)		//sub-type matrix
		{
			for (int ii=0; ii<n; ii++)
			{
				for(int iii=0; iii<n; iii++)
					fprintf(f6,"%.6lf\t",domain_mat[i][ii][iii]);
			}
			fprintf(f6,"\n");
		}
		fclose(f2);
		fclose(f3);
		fclose(f5);
		fclose(f6);
	}
}

//calc entity representation
void calc_entity_vec(int head, int tail, int rel, int tid)		//use Weighted Hierarchy Encoder
{
	int tempHeadType = head_type_vec[rel];
	int tempTailType = tail_type_vec[rel];
	int tempHeadDomain = head_domain_vec[rel];
	int tempTailDomain = tail_domain_vec[rel];
	//build head_final_vec
	for(int i=0; i<n; i++)
	{
		double type_score = 0, domain_score = 0;
		for(int ii=0; ii<n; ii++)
		{
			type_score += type_mat[tempHeadType][i][ii] * entity_vec[head][ii];
		}
		for(int ii=0; ii<n; ii++)
		{
			domain_score += domain_mat[tempHeadDomain][i][ii] * entity_vec[head][ii];
		}
		head_final_vec[tid][i] = type_score * type_weight + domain_score * (1-type_weight);
	}
	//build tail_final_vec
	for(int i=0; i<n; i++)
	{
		double type_score = 0, domain_score = 0;
		for(int ii=0; ii<n; ii++)
		{
			type_score += type_mat[tempTailType][i][ii] * entity_vec[tail][ii];
		}
		for(int ii=0; ii<n; ii++)
		{
			domain_score += domain_mat[tempTailDomain][i][ii] * entity_vec[tail][ii];
		}
		tail_final_vec[tid][i] = type_score * type_weight + domain_score * (1-type_weight);
	}
}

double calc_sum_triple(int e1,int e2,int rel, int flag, int tid)		//similarity
{
	double sum=0;
	calc_entity_vec(e1, e2, rel, tid);
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = tail_final_vec[tid][ii]-head_final_vec[tid][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = tail_final_vec[tid][ii]-head_final_vec[tid][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = tail_final_vec[tid][ii]-head_final_vec[tid][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = tail_final_vec[tid][ii]-head_final_vec[tid][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
}

void gradient_triple(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//SGD update
{
	int tempDomain = -1;
	int tempType = -1;
	double der_1 = -1;
	//positive triple
	//relation
	for(int i=0; i<n; i++)
	{
		relation_tmp[rel_a][i] += rate*posErrorVec[tid][i];
	}
	//head
	tempDomain = head_domain_vec[rel_a];
	tempType = head_type_vec[rel_a];
	for(int i=0; i<n; i++)
	{
		der_1 = rate*posErrorVec[tid][i]*(1-type_weight);
		for(int ii=0; ii<n; ii++)
		{
			domain_mat_tmp[tempDomain][i][ii] += der_1 * entity_vec[e1_a][ii];
			entity_tmp[e1_a][ii] += der_1 * domain_mat[tempDomain][i][ii];
		}
		der_1 = rate*posErrorVec[tid][i]*type_weight;
		for(int ii=0; ii<n; ii++)
		{
			type_mat_tmp[tempType][i][ii] += der_1 * entity_vec[e1_a][ii];
			entity_tmp[e1_a][ii] += der_1 * type_mat[tempType][i][ii];
		}
	}
	//tail
	tempDomain = tail_domain_vec[rel_a];
	tempType = tail_type_vec[rel_a];
	for(int i=0; i<n; i++)
	{
		der_1 = -rate*posErrorVec[tid][i]*(1-type_weight);
		for(int ii=0; ii<n; ii++)
		{
			domain_mat_tmp[tempDomain][i][ii] += der_1 * entity_vec[e2_a][ii];
			entity_tmp[e2_a][ii] += der_1 * domain_mat[tempDomain][i][ii];
		}
		der_1 = -rate*posErrorVec[tid][i]*type_weight;
		for(int ii=0; ii<n; ii++)
		{
			type_mat_tmp[tempType][i][ii] += der_1 * entity_vec[e2_a][ii];
			entity_tmp[e2_a][ii] += der_1 * type_mat[tempType][i][ii];
		}
	}
	//negative triple
	for (int i = 0;i<n;i++)
	{
		relation_tmp[rel_b][i] -= rate*negErrorVec[tid][i];
	}
	//head
	tempDomain = head_domain_vec[rel_b];
	tempType = head_type_vec[rel_b];
	for(int i=0; i<n; i++)
	{
		der_1 = -rate*negErrorVec[tid][i]*(1-type_weight);
		for(int ii=0; ii<n; ii++)
		{
			domain_mat_tmp[tempDomain][i][ii] += der_1 * entity_vec[e1_b][ii];
			entity_tmp[e1_b][ii] += der_1 * domain_mat[tempDomain][i][ii];
		}
		der_1 = -rate*negErrorVec[tid][i]*type_weight;
		for(int ii=0; ii<n; ii++)
		{
			type_mat_tmp[tempType][i][ii] += der_1 * entity_vec[e1_b][ii];
			entity_tmp[e1_b][ii] += der_1 * type_mat[tempType][i][ii];
		}
	}
	//tail
	tempDomain = tail_domain_vec[rel_b];
	tempType = tail_type_vec[rel_b];
	for(int i=0; i<n; i++)
	{
		der_1 = rate*negErrorVec[tid][i]*(1-type_weight);
		for(int ii=0; ii<n; ii++)
		{
			domain_mat_tmp[tempDomain][i][ii] += der_1 * entity_vec[e2_b][ii];
			entity_tmp[e2_b][ii] += der_1 * domain_mat[tempDomain][i][ii];
		}
		der_1 = rate*negErrorVec[tid][i]*type_weight;
		for(int ii=0; ii<n; ii++)
		{
			type_mat_tmp[tempType][i][ii] += der_1 * entity_vec[e2_b][ii];
			entity_tmp[e2_b][ii] += der_1 * type_mat[tempType][i][ii];
		}
	}
}

void train_triple_mul(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//margin-based score function
{
	double sum1 = calc_sum_triple(e1_a,e2_a,rel_a,1,tid);		//positive score
	double sum2 = calc_sum_triple(e1_b,e2_b,rel_b,0,tid);		//negative score
	if (sum1+margin>sum2)
	{
		res_thread_triple[tid]+=margin+sum1-sum2;
		gradient_triple( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void prepare()		//preprocessing
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	FILE* f3 = fopen("../data/type2id.txt","r");
	FILE* f4 = fopen("../data/domain2id.txt","r");
	FILE* f5 = fopen("../data/relationType.txt","r");
	FILE* f6 = fopen("../data/relationDomain.txt","r");
	FILE* f7 = fopen("../data/typeEntity.txt","r");
	int x, y, z;
	//build entity2ID、ID2entity map
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;		//<entity,ID>
		id2entity[x]=st;		//<ID,entity>
		entity_num++;
	}
	//build relation2ID、ID2relation map
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	//build type2id、id2type map
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		type2id[st]=x;		//<type,ID>
		id2type[x]=st;		//<ID,type>
		type_num++;
	}
	//build domain2id、id2domain map
	while (fscanf(f4,"%s%d",buf,&x)==2)
	{
		string st=buf;
		domain2id[st]=x;
		id2domain[x]=st;
		domain_num++;
	}
	//build relation-specific information for type
	head_type_vec.resize(relation_num);
	tail_type_vec.resize(relation_num);
	while (fscanf(f5,"%s",buf)==1)
	{
		int tempRel = relation2id[buf];
		fscanf(f5,"%s",buf);
		head_type_vec[tempRel] = type2id[buf];
		fscanf(f5,"%s",buf);
		tail_type_vec[tempRel] = type2id[buf];
	}
	//build relation-specific information for domain
	head_domain_vec.resize(relation_num);
	tail_domain_vec.resize(relation_num);
	while (fscanf(f6,"%s",buf)==1)
	{
		int tempRel = relation2id[buf];
		fscanf(f6,"%s",buf);
		head_domain_vec[tempRel] = domain2id[buf];
		fscanf(f6,"%s",buf);
		tail_domain_vec[tempRel] = domain2id[buf];
	}
	//build type_entity_list
	type_entity_list.resize(type_num);
	type_entity_num.resize(type_num);
	while (fscanf(f7,"%d%d",&y,&x)==2)
	{
		type_entity_num[y] = x;
		type_entity_list[y].resize(x);
		for(int i=0; i<x; i++)
		{
			fscanf(f7,"%d",&z);
			type_entity_list[y][i] = z;		//type->entity
		}
	}
	//read triple set
    FILE* f_kb = fopen("../data/train.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;		//head
        fscanf(f_kb,"%s",buf);
        string s2=buf;		//tail
        fscanf(f_kb,"%s",buf);
        string s3=buf;		//relation
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
	cout<<"domain_num="<<domain_num<<endl;
	cout<<"type_num="<<type_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int n = 50;		//dimention of entity/relation
	rate = 0.0025;		//learning rate
    rate_begin = 0.0025;		//begin
	rate_end = 0.0001;		//end
    double margin = 1;		//loss margin
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();
    run(n,rate,margin,method);
}


