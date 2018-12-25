0function [train_data,sift_feat] = Sift(images, imsize)

    fprintf('\nSift\n')

    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    step_p = 8; % Sampling density 
    binSize = 8; % Scale of sxtracted descriptors, pixel size of each bin of the histogram of each interest's point descriptor
    
    for i = 1:num
        img = images{i};
        img=imresize(img,imsize,'bilinear');
        if mod(i,1000) ==0
            disp(i);
        end
        img = single(rgb2gray(img));
        input_img = vl_imsmooth(img,0.5);
        [~, sift_features] = vl_dsift(input_img,'Step',step_p,'size', binSize,'fast');
        sift_features = single(sift_features)';
        [sift_feat,~] = size (sift_features);
        train_data = [train_data;sift_features];
    end 
    
end