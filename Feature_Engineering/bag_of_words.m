function image_feats = bag_of_words(features,codebook,features_per_image)
    vocab_size = size(codebook, 2);
    [~,num] = size(features);
    num = num/ (features_per_image);
    
    image_feats = zeros(num,vocab_size);
    j=1;
    for i = 1:num
        feature_vector = features(:,j:features_per_image-1+j);
        dist = vl_alldist2(feature_vector,codebook);

        [~,index]=min(dist,[],2);
        hist_v =histc(index,[1:1:vocab_size]);
        image_feats(i,:) = hist_v;
      
        if mod(i,1000) ==0
            fprintf('\n image %d \n',i);
        end
        j=features_per_image+j;
    end
end