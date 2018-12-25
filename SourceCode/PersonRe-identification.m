% Image and Visual Computing Assignment 1: Person Re-identification
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with person re-identification problem. The vl_feat, 
%   libsvm, liblinear and any other classification and feature extraction 
%   library are allowed to use in this assignment. The built-in matlab 
%   object-detection functionis not allowed. Good luck and have fun!
%
%                                               Released Date:   7/11/2018
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
%==========================================================================
clear all
clc
run ICV_setup

% Hyperparameter of experiments
resize_size=[128 64];


%% Part 1: Person Re-identification (re-id): 
%==========================================================================
% The aim of this task is to verify whether the two given people in the
% images are the same person. We train a binary classifier to predict
% whether these two people are actually the same person or not.
% - Extract the features
% - Get a data representation for training
% - Train the re-identifier and evaluate its performance
%==========================================================================


load('./data/person_re-identification/person_re-id_train.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% - train, a 1 x N struct array with fields image1, id1, image2 and id2
% - image1 and image2 are two image while id1 and id2 are the corresponding
% label. Here, we generate a label vector (i.e. Ytr) which indicates image1 
% is the same person as the image2 or not, where, Ytr = 1 indicates 'same' 
% and Ytr = -1 represents 'different'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% You should construct the features in here. (read, resize, extract)

image1 = {train(:).image1}';
image2 = {train(:).image2}';
id1 = [train(:).id1]';
id2 = [train(:).id2]';

Ytr = ones(length(id1),1);
Ytr(id1 ~= id2) = -1;
disp('Person Re-id:Extracting features..');


 
binsize = 25;
Lbp_neig = 4;
lbp_cell = 16;
% 

Xtr1_color = Color_Histogram(image1,resize_size,binsize);
Xtr2_color = Color_Histogram(image2,resize_size,binsize);

disp('1/2 ');

Xtr_lbp1 = LBP(image1,resize_size,Lbp_neig,lbp_cell);
Xtr_lbp2 = LBP(image2,resize_size,Lbp_neig,lbp_cell);

disp('2/2');

Xtr1 = [Xtr1_color,Xtr_lbp1];
Xtr2 = [Xtr2_color,Xtr_lbp2];

Xtr = double(abs(Xtr1 - Xtr2));


model = fitcsvm(Xtr, Ytr,'KernelFunction','rbf','KernelScale',20);
% model = fitcknn(Xtr,Ytr,'NumNeighbors',3);
% save('person_reid_model.mat','model');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the testing data
% - query, a 1 x N struct array with fields image and id
% - gallery, a 1 x N struct array with fiedls image and id
% - In the testing, we aim at re-identifying the query person in the
% gallery images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./data/person_re-identification/person_re-id_test.mat')

image_query = {query(:).image}';
id_query = [query(:).id]';
image_gallery = {gallery(:).image}';
id_gallery = [gallery(:).id]';
num_query = length(image_query);
num_gallery = length(image_gallery);

Xq = Color_Histogram(image_query,resize_size,binsize);
Xg = Color_Histogram(image_gallery,resize_size,binsize);

Xq_lbp1 = LBP(image_query,resize_size,Lbp_neig,lbp_cell);
Xq_lbp2 = LBP(image_gallery,resize_size,Lbp_neig,lbp_cell);


Xq = [Xq,Xq_lbp1];
Xg = [Xg,Xq_lbp2];


% Constructing query-gallery pairs and label vector
Xte = [];
Yte = [];
for i = 1:length(image_query)
    Xq_ = Xq(i,:);
    temp_Xte = abs(Xq_ - Xg);
    Xte = [Xte; temp_Xte];
    temp_Yte = ones(num_gallery,1);
    temp_Yte(id_gallery ~= id_query(i)) = -1;
    Yte = [Yte; temp_Yte];
end
Xte = double(Xte);


[l,prob] = predict(model,Xte);

% Compute the mean Average Precision

ap = zeros(num_query, 1);
for i = 1:num_query
    prob_i = prob((i - 1) * num_gallery + 1: i * num_gallery,2);
    [~, sorted_index] = sort(prob_i, 'descend');
    temp_index = 1:num_gallery;
    same_index = temp_index(id_gallery == id_query(i));
    [ap(i), ~] = compute_AP(same_index, sorted_index);
end
mAP = mean(ap);
fprintf('The mean Average Precision of person re-identification is:%.2f \n', mAP * 100)


%% Visualization the result of person re-id

%{
query_idx = [2,15,26]; 
gallery_idx = [2, 1, 45];
l_ = reshape(l(:), [90, 30])';
Yte_ = reshape(Yte, [90, 30])';
nPairs = 3; % number of visualize data. maximum is 3
% nPairs = length(data_idx); 
visualise_reid(image_query, image_gallery, query_idx, gallery_idx, l_, Yte_, nPairs )
%}

%% Last extracted features 

% [feat_sift_test1, sift_feat] = Sift(image_query,resize_size);
% [feat_sift2_test, ~] = Sift(image_gallery,resize_size);
% 
% [feat_te1,~] = Hog_LBP_CH(image_query, resize_size);
% [feat_te2,~] = Hog_LBP_CH(image_gallery, resize_size);

% feat_te1 = double(feat_sift_test1)';
% feat_te2 = double(feat_sift2_test)';

% Xq = bag_of_words(feat_te1,c1,sift_feat);
% Xg = bag_of_words(feat_te2,c2,sift_feat);
