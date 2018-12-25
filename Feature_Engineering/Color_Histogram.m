function [train_data] = Color_Histogram(images,imsize,n)
    num = length(images);
    train_data = []; % to hold all the HoG for training
    fprintf('\n Extracting Color histogram \n')

    for i = 1:num
        img = images{i};
        img=imresize(img,imsize,'bilinear');
%         if mod(i,400) ==0
%             disp(i);
%         end
        norm_factor = 350;
        norm_factor_part = 200;

        histR = (imhist(img(:,:,1),n)./norm_factor)';
        histG = (imhist(img(:,:,2),n)./norm_factor)';
        histB = (imhist(img(:,:,3),n)./norm_factor)';
        
        temp = [];
        for i=1:16:112
            for j = 1:8:48
                [y,yy,x,xx] = deal(i, i+16,j,j+8);
                hist= [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
                temp = [temp, hist];
            end
        end
        
      for i=4:16:112
            [y,yy,x,xx] = deal(i, i+16,10,54);
            hist= [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
            temp = [temp, hist];
      end
       
      
       
        
%         for i=4:16:486
%             [y,yy,x,xx] = deal(1, 128,i,i+16);
%             hist= [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
%             temp = [temp, hist];
%         end
        
        
        %coordinates for head
%         [y,yy,x,xx] = deal(1, 32,20,44);
%         histHead = [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
%         %coordinates for body
%         [y,yy,x,xx] = deal(33, 96,9,56);
%         histBody = [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
%         %coordinates for feet
%         [y,yy,x,xx] = deal(97, 128,1,64);
%         histFeet = [(imhist(img(y:yy,x:xx,1),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,2),n)./norm_factor_part)',(imhist(img(y:yy,x:xx,3),n)./norm_factor_part)'];
%         hist  = [histR,histG,histB,histHead,histBody,histFeet,temp];
        train_data = [train_data;temp];
    end
end

