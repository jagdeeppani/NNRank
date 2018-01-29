function Accuracy=find_accuracy_munkres(result,label,k)

n=length(result);
% find accuracy assumes there are k topics 1 to k, if that is not true. The following section takes care of this mapping the topic numbers in 'topic' [1,k]
if length(unique(result)) ~= length(unique(label))
    error('The clustering result is not consistent with label.');
end

if length(unique(result)) ~= k
    fprintf('There are classes without any points');
end


mapped_label = label;
unique_label = unique(label); % returns sorted unique gnd

for ug_count=1:k
    mapped_label(label == (unique_label(ug_count)) ) = ug_count;
end


topic_class = zeros(k,k);
for doc_idx=1:n
    temp1 = result(doc_idx);
    temp2 = mapped_label(doc_idx);
    topic_class(temp1,temp2) = topic_class(temp1,temp2)+1;
end

costMat = -1*topic_class;
[assignment,~] = munkres(costMat);

    intersect_sum = 0;
for i=1:k
    intersect_sum = intersect_sum +topic_class(i,assignment(i));
    
end

Accuracy = intersect_sum/n;
end

