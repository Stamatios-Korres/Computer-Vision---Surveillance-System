function  [train_data,features] = LBP(images, imsize,neighboors,cell_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    disp("bp features");
    num = length(images);
    train_data = []; % to hold all the HoG for training
    for i = 1:num
        img = images{i};
        img=imresize(img,imsize,'bilinear');

        img = rgb2gray(img);

        lbp = extractLBPFeatures(img,'Upright',false,'NumNeighbors',neighboors,'CellSize',[cell_size,cell_size]);
        train_data = [train_data;lbp];
    end 
    
    
    
end

