function [train_data,num_features] = HoG(images, imsize,cell_size)

    dir = pwd();
    num = length(images);
    train_data = []; % to hold all the HoG for training
    fprintf('\nLHog\n')

    for i = 1:num
        img = images{i};
        img=imresize(img,imsize,'bilinear');
        img = rgb2gray(img);
        [y,yy,x,xx] = deal(10, 70,22,38);
        img = img(y:yy,x:xx);
        hog = extractHOGFeatures(img,'CellSize',[cell_size cell_size]);
%         hog = reshape(hog,9,[])';
%         [num_features,~] = size(hog);
        train_data = [train_data;[hog]];
    end 
    
end