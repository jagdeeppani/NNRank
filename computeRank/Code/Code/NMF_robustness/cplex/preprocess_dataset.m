function fea = preprocess_dataset(fea, R_terms, stop_words)

%stop_words=importdata('stopwords.txt');
%data1=importdata('Reuter_terms.txt');
%R_terms=data1.textdata;
%clear data1;

voc_size = length(R_terms);
no_of_sws=length(stop_words);

count=1;
for i=1:voc_size
    status = str2double(R_terms{i});
    if ~isnan(status)
        remove_idx(count)=i;
        count=count+1;
    end
end

for i=1:no_of_sws
    index = find(strcmp(stop_words{i},R_terms));
    if ~isempty(index)
        remove_idx(count)=index;
        count=count+1;
    end
end

fea(:,remove_idx)=[];
end

        