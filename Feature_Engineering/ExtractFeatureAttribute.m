function [train_data,label] = ExtractFeatureAttribute(images, imsize, class)

    if nargin <= 2
        class = 'None';
        label = [];
    end
    
    dir = pwd();
    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    for i = 1:num
        img = images{i};
        img=imresize(img,imsize,'bilinear');
        normaliser = 300;
        n = 20;
        [y,yy,x,xx] = deal(1, 32,16,48);
        histHead = [(imhist(img(y:yy,x:xx,1),n)./normaliser)',(imhist(img(y:yy,x:xx,2),n)./normaliser)',(imhist(img(y:yy,x:xx,3),n)./normaliser)'];
        [y,yy,x,xx] = deal(33, 96,9,56);
        histBody = [(imhist(img(y:yy,x:xx,1),n)./normaliser)',(imhist(img(y:yy,x:xx,2),n)./normaliser)',(imhist(img(y:yy,x:xx,3),n)./normaliser)'];
        [y,yy,x,xx] = deal(97, 128,1,64);
        histFeet = [(imhist(img(y:yy,x:xx,1),n)./normaliser)',(imhist(img(y:yy,x:xx,2),n)./normaliser)',(imhist(img(y:yy,x:xx,3),n)./normaliser)'];
        img = rgb2gray(img);
        
        %hog = extractHOGFeatures(img,'CellSize',[16 16]);
        [y,yy,x,xx] = deal(1, 32,16,48);
        hogHead = extractHOGFeatures(img(y:yy,x:xx),'CellSize',[16,16]);
        [y,yy,x,xx] = deal(33, 96,9,56);
        hogBody = extractHOGFeatures(img(y:yy,x:xx),'CellSize',[16,16]);
        [y,yy,x,xx] = deal(97, 128,1,64);
        hogFeet = extractHOGFeatures(img(y:yy,x:xx),'CellSize',[16,16]);
        lbp = extractLBPFeatures(img,'CellSize',[32 32]);
        tmp = [hogFeet,hogBody,hogHead,histHead,histBody,histFeet,lbp];
        train_data = [train_data;tmp]; 
        disp(i/num);
    end 
    
    num = size(train_data,1);
    if strcmp(class, 'pos')
        label = ones(num,1);
    end
    if strcmp(class, 'neg')
        label = zeros(num,1);
    end