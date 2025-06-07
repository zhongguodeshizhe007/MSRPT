clear variables;close all;clc
load data.mat % load results of the survey per participant (Table T)
Xi=NaN(210,3);
ini % load vehicle speed info, road type (1, 2, or 3), and titles of the 80 labels
load yoloresults % provides results of the YOLOv4 algorithm for the 210 images. cell BB, LL, and SS
% BB - Bounding boxes (x and y coordinates of left top, width, height) 
% SS - Classification scores
% LL - Class labels

F=fields(T);

temp=T.have_you_read_and_understood_the_above_instructions;X(:,1)=strcmp(temp,'no')+2*strcmp(temp,'yes');
temp=T.what_is_your_gender;X(:,2)=1*strcmp(temp,'female')+2*strcmp(temp,'male')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.what_is_your_age;X(:,3)=temp;X(X(:,3)>110,3)=NaN; % People who report age greater than 110 years
temp=T.about_how_many_kilometers_miles_did_you_drive_in_the_last_12_mo;for i=1:length(temp);try X(i,4)=1+cell2mat(temp(i));catch error;X(i,4)=1*strcmp(temp(i),'0_km__mi')+2*strcmp(temp(i),'1__1000_km_1__621_mi')+3*strcmp(temp(i),'1001__5000_km_622__3107_mi')+4*strcmp(temp(i),'5001__15000_km_3108__9321_mi')+5*strcmp(temp(i),'15001__20000_km_9322__12427_mi')+6*strcmp(temp(i),'20001__25000_km_12428__15534_mi')+7*strcmp(temp(i),'25001__35000_km_15535__21748_mi')+8*strcmp(temp(i),'35001__50000_km_21749__31069_mi')+9*strcmp(temp(i),'50001__100000_km_31070__62137_mi')+10*strcmp(temp(i),'more_than_100000_km_more_than_62137_mi')-1*strcmp(temp(i),'i_prefer_not_to_respond');end;end
temp=T.on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_mon;X(:,5)=1*strcmp(temp,'never')+2*strcmp(temp,'less_than_once_a_month')+3*strcmp(temp,'once_a_month_to_once_a_week')+4*strcmp(temp,'1_to_3_days_a_week')+5*strcmp(temp,'4_to_6_days_a_week')+6*strcmp(temp,'every_day')-1*strcmp(temp,'i_prefer_not_to_respond');
X(:,6)=T.at_which_age_did_you_obtain_your_first_license_for_driving_a_ca;
X(:,7)=datenum(T.x_started_at);
X(:,8)=datenum(T.x_created_at);
X(:,9)=round(2400*36*(X(:,8) - X(:,7)));
temp=T.how_often_do_you_do_the_following_using_a_mobile_phone_without_;X(:,10)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_driving_so_close_to_the_car_i;X(:,11)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_sounding_your_horn_to_indicat;X(:,12)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_becoming_angered_by_a_particu;X(:,13)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_racing_away_from_traffic_ligh;X(:,14)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_disregarding_the_speed_limi_1;X(:,15)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
temp=T.how_often_do_you_do_the_following_disregarding_the_speed_limit_;X(:,16)=1*strcmp(temp,'0_times_per_month')+2*strcmp(temp,'1_to_3_times_per_month')+3*strcmp(temp,'4_to_6_times_per_month')+4*strcmp(temp,'7_to_9_times_per_month')+5*strcmp(temp,'10_or_more_times_per_month')-1*strcmp(temp,'i_prefer_not_to_respond');
% 1. using a mobile phone without a hands free kit
% 2. driving so close to the car in front that it would be difficult to stop in an emergency
% 3. sounding the horn to indicate annoyance with another road user
% 4. becoming angered by a particular type of driver, and indicate hostility by whatever means one can
% 5. racing away from traffic lights with the intention of beating the driver next to own vehicle;
% 6. disregarding the speed limit on a residential road
% 7. disregarding the speed limit on a motorway;
X(X<0)=NaN;
X(isnan(sum(X(:,10:16),2)),10:16)=NaN; % for the DBQ put all 7 items at NaN if any of the 7 items is NaN
X(:,17)=mean(X(:,15:16),2); % Speeding violatins
X(:,18)=mean(X(:,10:14),2); % Non-speeding violations
temp=T.what_is_your_primary_mode_of_transportation;X(:,19)=1*strcmp(temp,'private_vehicle')+2*strcmp(temp,'public_transportation')+3*strcmp(temp,'motorcycle')+4*strcmp(temp,'walkingcycling')+5*strcmp(temp,'other')-1*strcmp(temp,'i_prefer_not_to_respond');
X(:,20)=T.how_many_accidents_were_you_involved_in_when_driving_a_car_in_t;

%% Extract response, response time, and trial number in 210 images x participants matrix
[R,Rt,Rti]=deal(NaN(210,size(X,1)));
for i=1:210 % loop over 210 images
    id=strcmp(F,['Stimulus' num2str(i-1) '_Response']); % response
    R(i,:)=T.(F{id});
    id=strcmp(F,['Stimulus' num2str(i-1) '_Rt']); % response time
    Rt(i,:)=T.(F{id});
    id=strcmp(F,['Stimulus' num2str(i-1) '_Trial_index']); % trial number (order of presentation) (2, 4, 6, ... 200)
    Rti(i,:)=T.(F{id});
end

%% Create 100 x participants matrix with response and response time in chronological order
[Ro,Rto]=deal(NaN(100,size(X,1)));
for i=1:size(X,1) % loop over participants
    for i2=2:2:200 % loop over trial numbers
        id=find(Rti(:,i)==i2); % find trial number
        if ~isempty(id)
            Ro(floor(i2/2),i)=R(id,i); % store response
            Rto(floor(i2/2),i)=Rt(id,i); % store response time
        end
    end
end
%% Remove participants who did not meet the criteria
invalid1 = find(X(:,1)==1); % respondents who indicated that they did not read instructions
invalid2 = find(X(:,3)<18); % respondents who indicated they are under 18 years old
invalid3 = find(X(:,9)<300); % respondents who took less than 5 min to complete the entire study
invalid5 = find(sum(~isnan(Rto))<90)'; % respondents with fewer than 90 responses
%% Find rows with identical IP addresses
y = NaN(size(X(:,1)));
IP=T.x_ip;
IPCF_1=NaN(1,size(X,1));
for i=1:size(X,1)
    try IPCF_1(i)=str2double(strrep(IP(i),'.',''));
    catch
        IPCF_1(i)=cell2mat(IP(i));
    end
end % reduce IP addresses of appen data to a single number
for i=1:size(X,1)
    temp=find(IPCF_1==IPCF_1(i));
    if length(temp)==1 % if the IP address occurs only once
        y(i)=1; % keep
    elseif length(temp)>1 % if the IP address occurs more than once
        y(temp(1))=1; % keep the first survey for that IP address
        y(temp(2:end))=2; % do not keep the other ones
    end
end
invalid4=find(y>1); % respondents who completed the survey more than once based on IP address (i.e., remove the doublets)
invalid6=find(std(R,'omitnan')<5)';
%% Remove invalid participants
invalid = unique([invalid1;invalid2;invalid3;invalid4;invalid5;invalid6]); % combine all invalid participants into one vector
R(:,invalid)=[];
Rt(:,invalid)=[];
Rti(:,invalid)=[];
Ro(:,invalid)=[];
Rto(:,invalid)=[];
T(invalid,:)=[];
X(invalid,:)=[];
%% Output general stats about the participants
disp([datestr(now, 'HH:MM:SS.FFF') ' - Survey time mean (minutes) - Before filtering = ' num2str(mean(X(:,9)/60,'omitnan'))]);
disp([datestr(now, 'HH:MM:SS.FFF') ' - Survey time median (minutes) - Before filtering = ' num2str(median(X(:,9)/60,'omitnan'))]);
disp([datestr(now, 'HH:MM:SS.FFF') ' - Survey time SD (minutes) - Before filtering = ' num2str(std(X(:,9)/60,'omitnan'))]);
disp([datestr(now, 'HH:MM:SS.FFF') ' - First survey start date - Before filtering = ' datestr(min(X(:,7)))]);
disp([datestr(now, 'HH:MM:SS.FFF') ' - Last survey end date - Before filtering = ' datestr(max(X(:,8)))]);
disp([datestr(now, 'HH:MM:SS.FFF') ' - Gender, male = ' num2str(sum(X(:,2)==2))])
disp([datestr(now, 'HH:MM:SS.FFF') ' - Gender, female = ' num2str(sum(X(:,2)==1))])
disp([datestr(now, 'HH:MM:SS.FFF') ' - Gender, I prefer not to respond = ' num2str(sum(isnan(X(:,2))))])
disp([datestr(now, 'HH:MM:SS.FFF') ' - Age, mean = ' num2str(mean(X(:,3),'omitnan'))])
disp([datestr(now, 'HH:MM:SS.FFF') ' - Age, sd = ' num2str(std(X(:,3),'omitnan'))])

% Most common countries (after filtering)
[~, ~, ub] = unique(T.x_country);
test2counts = histcounts(ub, 'BinMethod','integers');
[B,I] = maxk(test2counts,10);
country_unique = unique(T.x_country);
disp([datestr(now, 'HH:MM:SS.FFF') ' - Most common countries (after filtering): '])
disp(country_unique(I)')
disp(B)
%%
RT=median(Rt,2,'omitnan'); % median response time per image
c=corr(R,mean(R,2,'omitnan'),'rows','pairwise'); % correlation between mean perceived risk and participant perceived risk

disp('Section 3.1 - Mean number of ratings per participant')
disp(round(mean(sum(~isnan(R))),1))
disp('Section 3.1 - SD number of ratings per participant')
disp(round(std(sum(~isnan(R))),2))

disp('Section 3.1 - Mean number of ratings per image')
disp(round(mean(sum(~isnan(R'))),1))
disp('Section 3.2 - SD number of ratings per image')
disp(round(std(sum(~isnan(R'))),2))

LLn=NaN(210,15,80);
for i=1:210 % loop over 210 images
    for l=1:80 % loop over 80 classes
        if ~isempty(LL{i}) % if there are labels of image
            id=find(LL{i}==Labels{l}); % find image labels that corresponding to label number l
            if ~isempty(id) % if there are such labels
                LLn(i,id,l)=l; % store label number l
            end
        end
    end
end

%% Figure 3. Mean perceived risk per image as reported by even-numbered participants (n = 689) versus mean perceived risk as reported by odd-numbered participants (n = 689). Each marker represents an image and is based on an average of 326 responses.
v1=R(:,1:2:end);
v2=R(:,2:2:end);
disp('Figure 3: sample sizes')
disp([size(v1,2) size(v2,2)])
disp('Figure - Number of ratings per marker')
disp(round([mean(sum(~isnan(v1),2)) mean(sum(~isnan(v2),2))],2))
v1=mean(v1,2,'omitnan'); % PPR - Odd-numbered participants (%)
v2=mean(v2,2,'omitnan'); % PPR - Even-numbered participants (%)
disp('Section 3.2: Correlation coefficient belonging to Figure 3')
disp(round(corr(v1,v2),2))
figure
s=scatter(v1,v2,140,'markerfacecolor',[0 0 1],'markeredgecolor',[0 0 1]);hold on;grid on
s.MarkerFaceAlpha = .3;
hold on
plot([0 65],[0 65],'k')
axis equal
xlabel('PPR - Odd-numbered participants (%)')
ylabel('PPR - Even-numbered particpants (%)')
h=findobj('FontName','Helvetica'); set(h,'FontSize',24,'Fontname','Arial');
set(gca,'LooseInset',[0.01 0.01 0.01 0.01]);
%% Section 3.2. Further exploration at the level of participants (n = 1378) showed no strong associations (|r| < 0.06) between participants’ mean risk across the rated images and gender (1 = female, 2 = male), age, driving mileage in the past 12 months (rated on a scale from 1 = 0 km to 10 = more than 100,000 km), and driving frequency in the past 12 months (rated on a scale from 1 = never to 6 = every day).
disp('Further exploration at the level of participants (n = 1378) showed no strong associations (|r| < 0.06) between participants’ mean risk across the rated images and gender (1 = female, 2 = male), age, driving mileage in the past 12 months (rated on a scale from 1 = 0 km to 10 = more than 100,000 km), and driving frequency in the past 12 months (rated on a scale from 1 = never to 6 = every day).')
disp(round(corr(mean(R,'omitnan')',X(:,2:5),'rows','pairwise'),3))
%% Figure 4. Mean perceived risk per image for participants who reported driving less than once a month (n = 281) versus participants who reported driving once a month or more frequently (n = 1079). Each marker represents an image and is based on an average of 133 and 511 responses for the former and latter groups of participants.
v1=R(:,X(:,5)<=2);
v2=R(:,X(:,5)>2);
disp('Figure 4: sample sizes')
disp([size(v1,2) size(v2,2)])
disp('Figure 4: Number of ratings per marker')
disp(round([mean(sum(~isnan(v1),2)) mean(sum(~isnan(v2),2))]))
v1=mean(v1,2,'omitnan'); % PPR - Participants driving less than once a month (%)
v2=mean(v2,2,'omitnan'); % PPR - Participants driving once a month or more (%)
disp('Section 3.2: Correlation coefficient belonging to Figure 4')
disp(round(corr(v1,v2),2))
figure
s=scatter(v1,v2,140,'markerfacecolor',[0 0 1],'markeredgecolor',[0 0 1]);hold on;grid on
s.MarkerFaceAlpha = .3;
hold on
plot([0 65],[0 65],'k')
axis equal
xlabel('PPR - Participants driving less than once a month (%)')
ylabel('PPR - Participants driving once a month or more (%)')
h=findobj('FontName','Helvetica'); set(h,'FontSize',24,'Fontname','Arial');
set(gca,'LooseInset',[0.01 0.01 0.01 0.01]);
%% Section 3.2: PPR per country
disp('Section 3.2: An examination of perceived risk averaged across all images showed differences in mean risk for participants from')
for i=1:5
    disp(country_unique(I(i)))
    disp([round(mean(mean(R(:,ismember(T.x_country,country_unique(I(i)))),2,'omitnan')),1) sum(ismember(T.x_country,country_unique(I(i)))) ])
end
%% Figure 5. Mean perceived risk per image for participants from Ukraine and Russia combined (n = 132) versus participants from the USA (n = 95). Each marker represents an image and is based on an average of 63 and 45 responses for the former and latter groups of participants.
v1=R(:,ismember(T.x_country,'USA'));
v2=R(:,ismember(T.x_country,'UKR')|ismember(T.x_country,'RUS'));
disp('Figure 5 - sample sizes')
disp([size(v1,2) size(v2,2)])
disp('Number of ratings per marker')
disp(round([mean(sum(~isnan(v1),2)) mean(sum(~isnan(v2),2))]))
v1=mean(v1,2,'omitnan'); % PPR - Participants from the USA (%)
v2=mean(v2,2,'omitnan');% PPR - Participants from Ukraine or Russia (%)
disp('Section 3.2: Correlation coefficient belonging to Figure 5')
disp(round(corr(v1,v2),2))
figure
s=scatter(v1,v2,140,'markerfacecolor',[0 0 1],'markeredgecolor',[0 0 1]);hold on;grid on
s.MarkerFaceAlpha = .3;
hold on
axis equal
plot([0 65],[0 65],'k')
xlabel('PPR - Participants from the USA (%)')
ylabel('PPR - Participants from Ukraine or Russia (%)')
h=findobj('FontName','Helvetica'); set(h,'FontSize',24,'Fontname','Arial');
set(gca,'LooseInset',[0.01 0.01 0.01 0.01]);
%% Section 3.2: Bootstrapping simulation
reps=10000;
cb=NaN(reps,1);
for i=1:reps % loop over repititions
    i1=ceil(size(R,2)*rand(132,1)); % random 132 participants
    z=zeros(size(R,2),1);z(i1)=1;
    vi=find(z==0); % vi are the non-selected participants
    i2=vi(ceil(size(vi,1)*rand(95,1))); % random 95 participants
    v1=mean(R(:,i1),2,'omitnan');  % PPR for a random 132 participants
    v2=mean(R(:,i2),2,'omitnan');  % PPR for a random 95 participants
    cb(i)=corr(v1,v2); % correlation of PPR at the level of images
end
disp('According to a bootstrapping analysis with two groups of sizes 95 and 132, respectively, the mean (SD) correlation coefficient of the PPR ratings of the two images between the two groups was found to be')
disp([round(mean(cb),2) round(std(cb),3)])
%%
IM=squeeze(mean(LLn,3,'omitnan')); % Image class numbers for each of the 210 images (210 x 15 matrix)
classes_to_use=[1 2 3 4 6 7 8];
for i=1:210 % loop over images
    bboxes=BB{i}; % bounding box info of
    im=IM(i,:); % numbers of classes in image
    id=find(ismember(im,classes_to_use)); % indexes of classes to use
    if ~isempty(id)
        Xi(i,2)=mean(sqrt(bboxes(id,3).*bboxes(id,4))); % Area score
    else % if no classes, then Area score is set to 0
        Xi(i,2)=0;
    end
end

Xi(:,1)=sum(sum(sign(LLn(:,:,1)),3,'omitnan'),2); % number of persons (Class 1)
PR=mean(R,2,'omitnan'); % mean perceived risk of the images (n = 210)
y=PR; % mean perceived risk (n = 210)

for ra=1:2 % loop across two regression analyses (Table 3 and Table 4)
    if ra==1
        x=Xi(:,1:2); % excluding speed
    elseif ra==2
        x=Xi(:,1:3); % including speed
    end

    warning off all
    stu=regstats(y, x); % linesr regression analysis, unstandardized variables
    st=regstats(zscore(y), zscore(x)); % lienar regression analysis, standardized variabels
    warning on all
    if ra==1
        disp('Table 3. Regression analysis results for predicting population perceived risk (PPR) from computer-vision variables (n = 210).')
    elseif ra==2
        disp('Table 4. Regression analysis results for predicting population perceived risk (PPR) from computer-vision variables, including speed (n = 210).')
    end
    disp([round([stu.beta st.beta st.tstat.t ],2),round(1000*st.tstat.pval(1:end))/1000])
    disp(['Number of trials in the regression analysis = ' num2str(length(y))])
    disp('df, dfe, F, p, r, r-squared')
    disp([st.fstat.dfr st.fstat.dfe round(st.fstat.f,1) round(st.fstat.pval,3) round(corr(st.yhat,y),2) round(st.rsquare,2)])

    disp('Percentile relative to participants predicting the population risk')
    disp(round(100*(1-sum(c>corr(st.yhat,y))/length(c)),1))
end
%% Table 2. Correlation matrix of the two computer-vision features, vehicle speed, population perceived risk (PPR), road type, and population median response time (n = 210).
v=[Xi PR RType RT];
disp('Mean, SD, Correlation matrix')
disp([round([mean(v)' std(v)'],2) round(corr(v),2)])
%% Figure 7. True perceived risk (i.e., mean perceived risk per image) versus predicted perceived risk (using the regression model depicted in Table 4). Each marker represents an image.
figure
s=scatter(stu.yhat,PR,140,'markerfacecolor',[0 0 1],'markeredgecolor',[0 0 1]);hold on;grid on
s.MarkerFaceAlpha = .3;
hold on
plot([0 65],[0 65],'k')
axis equal
xlabel('Predicted PPR (%)')
ylabel('True PPR (%)')
h=findobj('FontName','Helvetica'); set(h,'FontSize',24,'Fontname','Arial');
set(gca,'LooseInset',[0.01 0.01 0.01 0.01]);
%% Figure 9. Distribution of observed correlation coefficients between participants’ (n = 1378) perceived risk of the 100 images they rated and the population perceived risk (PPR) of the same images.
opengl hardware
hc=histcounts(c,-1:0.01:1);
figure;
bar(0.005+(-1:0.01:0.99),hc,'facecolor',[.4 .4 .4])
grid on
xlabel('Correlation coefficient')
ylabel('Number of participants')
h=findobj('FontName','Helvetica'); set(h,'FontSize',24,'Fontname','Arial');
set(gca,'LooseInset',[0.01 0.01 0.01 0.01],'xlim',[-0.5 1],'xtick',-1:0.1:1);
disp('The mean correlation coefficient was')
disp(round(mean(c),2))
disp('with 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles of')
disp(round([prctile(c,1) prctile(c,5) prctile(c,25) prctile(c,50) prctile(c,75) prctile(c,95) prctile(c,99)],2))
%% Table 1. Mean and standard deviation (SD) of the number of class instances, proportion of images with the class instance, and Pearson correlation coefficient between the number of class instances and population perceived risk (PPR) (n = 210).
Classes_for_Table1=[1 2 3 4 6 7 8 10 12];
disp(Labels(Classes_for_Table1)); % display classes used in the analysis
a=squeeze(sum(sign(LLn(:,:,Classes_for_Table1)),2,'omitnan'));
disp('Number of objects per class, for the 210 images')
disp(a)
disp([round(mean(a)',2) round(std(a)',2) round(mean(a>0)',2) round(corr(a,PR),2)])
%% Figure 6. Eight of the 210 images, sorted from the lowest to the highest perceived risk.
v0=mean(R,2,'omitnan');
V=[1 30 60 90 120 150 180 210];
[d,o]=sort(v0);
disp('Images number and risk')
disp([o(V) round(d(V),1) Xi(o(V),1) round(Xi(o(V),2)) round(Xi(o(V),3),1)])
