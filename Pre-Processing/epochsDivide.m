function [U,R,D,L]=epochsDivide(M,trigger) %M is the EEG recording matrix [#channels, time points]

% tasks counter
upcount=sum(trigger.type==0);
rightcount=sum(trigger.type==1);
downcount=sum(trigger.type==2);
leftcount=sum(trigger.type==3);

%epochs counter
iu=1;
ir=1;
id=1;
il=1;

% epochs' matrices= [#epochs, #channels, time points]
U=zeros(upcount,size(M,1),max(trigger.duration));
R=zeros(rightcount,size(M,1),max(trigger.duration));
D=zeros(downcount,size(M,1),max(trigger.duration));
L=zeros(leftcount,size(M,1),max(trigger.duration));

for i=1:length(trigger.latency)
    % if up trigger starts, put epoch in U -> 9Hz
    if trigger.type(i)==0
        U(iu,:,:)=M(:,trigger.latency(i):trigger.latency(i)+trigger.duration(i)-1);
        iu=iu+1; %update counter
    % if right trigger starts, put epoch in R -> 10Hz
    elseif trigger.type(i)==1
        R(ir,:,:)=M(:,trigger.latency(i):trigger.latency(i)+trigger.duration(i)-1);
        ir=ir+1; 
    % if down trigger starts, put epoch in D -> 12Hz
    elseif trigger.type(i)==2
        D(id,:,:)=M(:,trigger.latency(i):trigger.latency(i)+trigger.duration(i)-1);
        id=id+1; 
    % if left trigger starts, put epoch in L -> 15Hz
    elseif trigger.type(i)==3
        L(il,:,:)=M(:,trigger.latency(i):trigger.latency(i)+trigger.duration(i)-1);
        il=il+1; 
    end
end