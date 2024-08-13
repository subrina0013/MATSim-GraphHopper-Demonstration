from xml.etree.ElementTree import ElementTree
import pandas as pd
import re
from core.gh_output import gh_output_file
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import gzip
import shutil
import scipy.stats as stats


def unzip_gz_file(input_file, output_file):
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

input_gz_file = 'MATSim_DMV\\output\\output_events.xml.gz'
output_file = 'data/output_events.xml'


def df_events():
    agent_list=[]
    time_list=[]
    link_list=[]
    position_list=[]
    types=[]
    from_list=[]
    to_list=[]
    tot_line=0

    unzip_gz_file(input_gz_file, output_file)

    for line in open("data/output_events.xml"):
        if 'type="actend" person="' in line and 'facility="' in line:
            person =  re.findall(r'person="(.*?)"', line)[0]
            time = re.findall(r'<event time="(.*?)"', line)[0]
            typ= re.findall(r'type="(.*?)"', line)[0]
            fr =  re.findall(r'facility="(.*?)"', line)[0]
            to =  ''
            link = re.findall(r'link="(.*?)"', line)[0]

            agent_list.append(person)
            time_list.append(time)
            from_list.append(fr)
            to_list.append(to)
            link_list.append(link)
            position_list.append(typ)


        if 'type="departure" person="' in line and 'legMode="car"' in line:
            person =  re.findall(r'person="(.*?)"', line)[0]
            time = re.findall(r'<event time="(.*?)"', line)[0]
            link = re.findall(r'link="(.*?)"', line)[0]
            agent_list.append(person)
            time_list.append(time)
            from_list.append('')
            to_list.append('')
            link_list.append(link)
            position_list.append('departure')

        if 'type="vehicle enters traffic" person="' in line and 'networkMode="car" relativePosition=' in line and '_bus' not in line:
            person =  re.findall(r'person="(.*?)"', line)[0]
            time = re.findall(r'<event time="(.*?)"', line)[0]
            link = re.findall(r'link="(.*?)"', line)[0]
            agent_list.append(person)
            time_list.append(time)
            from_list.append('')
            to_list.append('')
            link_list.append(link)
            position_list.append('start')

        if 'type="entered link"' in line and 'vehicle="' in line and '_bus' not in line:
            person =  re.findall(r'vehicle="(.*?)"', line)[0]
            time = re.findall(r'<event time="(.*?)"', line)[0]
            link = re.findall(r'link="(.*?)"', line)[0]
            agent_list.append(person)
            time_list.append(time)
            from_list.append('')
            to_list.append('')
            link_list.append(link)
            position_list.append('mid')

        if 'type="actstart" person="' in line and 'facility="' in line:
            person =  re.findall(r'person="(.*?)"', line)[0]
            time = re.findall(r'<event time="(.*?)"', line)[0]
            typ= re.findall(r'type="(.*?)"', line)[0]
            fr =  ''
            to =  re.findall(r'facility="(.*?)"', line)[0]
            link = re.findall(r'link="(.*?)"', line)[0]

            agent_list.append(person)
            time_list.append(time)
            from_list.append(fr)
            to_list.append(to)
            link_list.append(link)
            position_list.append(typ)

    dfevents= pd.DataFrame({'agents': agent_list, 'real_time': time_list, 'link_id': link_list, 'position': position_list,'from_facility':from_list, 'to_facility':to_list})
    dfevents[['agents','real_time','link_id']] = dfevents[['agents','real_time','link_id']].astype(float).astype('int64')
    dfevents.fillna('0', inplace=True)
    print('dfevents working.................')
    dfevents.to_csv('data/dfevents.csv',index=False)
    # print(dfevents.head(1))
    return dfevents

def dfStep():
    dfEvents = df_events()
    print('dfEvents for dfstep')
    print(dfEvents.head(1))

    data_frames = list(dfEvents.groupby('agents'))
    for agent, agent_data in data_frames:
        start_pos_l    = agent_data[agent_data.position=='start'].index.tolist()
        dept_pos_l     = agent_data[agent_data.position=='departure'].index.tolist()
        actend_pos_l   = agent_data[agent_data.position=='actend'].index.tolist()
        actstart_pos_l = agent_data[agent_data.position=='actstart'].index.tolist()
        mid_pos_l      = agent_data[agent_data.position=='mid'].index.tolist()

        for s in range(len(start_pos_l)):
            sp= start_pos_l[s]
            actE =max(i for i in actend_pos_l if i <= sp)
            actE_time = agent_data.loc[actE,'real_time']
            actE_fac = agent_data.loc[actE,'from_facility']
            agent_data.loc[agent_data.index == sp, 'from_facility'] = actE_fac

            agent_data.loc[agent_data.index == sp, 'new_time'] = actE_time

            if len(actstart_pos_l) >0:
                if actstart_pos_l[-1] >= sp:
                    actS =min(i for i in actstart_pos_l if i >= sp)
                    actS_fac = agent_data.loc[actS,'to_facility']
                    agent_data.loc[agent_data.index == sp, 'to_facility'] = actS_fac

                    mid_l =[i for i in mid_pos_l if i > sp and i < actS]
                    if len(mid_l) > 0:
                        agent_data.loc[agent_data.index.isin(mid_l), 'from_facility'] = actE_fac
                        agent_data.loc[agent_data.index.isin(mid_l), 'to_facility'] = actS_fac
                        agent_data.loc[agent_data.index.isin(mid_l), 'new_time'] = actE_time

        for acs in range(len(actstart_pos_l)):
            actS = actstart_pos_l[acs]
            actE =max(i for i in actend_pos_l if i <= actS)
            agent_data.loc[agent_data.index == actE, 'to_facility'] = agent_data.loc[actS,'to_facility']
            agent_data.loc[agent_data.index == actE, 'new_time'] = agent_data.loc[actE,'real_time']

    
    dfStep1 = pd.concat([df for _,df in data_frames])
    print('dfStep1 creating')
    dfStep1 = dfStep1[(dfStep1.position == 'start') | (dfStep1.position == 'mid') | (dfStep1.position == 'actend')]
    dfStep1.fillna(0, inplace=True)
    dfStep1=dfStep1[(dfStep1.new_time != 0) &(dfStep1.to_facility != '0')]
    dfStep1[['from_facility','to_facility']] = dfStep1[['from_facility','to_facility']].astype(str)
    dfStep1['from_to'] = dfStep1['from_facility'] + '_' + dfStep1['to_facility']
    dfStep1 = dfStep1[dfStep1.from_facility != dfStep1.to_facility]

    dfStep2 = dfStep1.groupby(['agents','from_to','new_time']).position.count().reset_index()
    dfStep2.drop(['position'], axis=1, inplace=True)
    dfStep2['from_to_ser'] =1
    dfStep2 = dfStep2.sort_values(by=['agents','new_time'])
    dfStep2['from_to_ser'] = dfStep2.groupby(['agents','from_to']).from_to_ser.cumsum()

    dfStep1.drop(['from_facility','to_facility'], axis=1, inplace=True)
    dfStep1= dfStep1.merge(dfStep2, on=['agents','from_to','new_time'], how='left')

    df_plan_faci2 = plan_fac()
    df2 = dfStep1.merge(df_plan_faci2[['agents','hr','departure_time_sec', 'from_to','from_to_ser']], on=['agents','from_to','from_to_ser']).fillna(0)

    link_osmid = pd.read_csv('data\\link_osmid_dmv_street.csv')
    df2.link_id = df2.link_id.astype('int64')
    df2= df2.merge(link_osmid[['link_id','osm_id','link_length']], on='link_id', how='left').fillna(0)

    df2=df2[(df2.position == 'start')|(df2.position == 'mid')]
    df2 =  df2.sort_values(['agents','real_time','from_to'])
    df2['trip_time'] = df2['real_time'] - df2.groupby(['agents','departure_time_sec'])['real_time'].transform('first')

    restricted_links = pd.read_csv('data/restricted_links.csv') ## copy again from G:/simulation........
    df2_restricted = df2[df2.osm_id.isin(restricted_links.osm_id.unique())].groupby(by=['agents','departure_time_sec','from_to','from_to_ser']).size().reset_index(name='counts')
    df2_restricted.to_csv('data/df2_restricted.csv', index=False)
    df2 = df2.merge(df2_restricted, on=['agents','departure_time_sec','from_to','from_to_ser'], how='left').fillna(0)
    df2 = df2[df2['counts'] == 0]
    df2.drop(['counts'], axis=1, inplace=True)

    # df2[['agents','real_time','departure_time_sec','from_to','from_to_ser','hr','link_id','osm_id','link_length','trip_time']].to_csv('data/df2_matsim.csv', index=False)
    print('df2 working ......................................')
    return df2


def mat_gh_length_time():
    df2 = dfStep()
    gh_output, gh_length_time = gh_output_file()
    mat_length2 = df2.groupby(['agents','departure_time_sec','from_to','from_to_ser','hr'])['link_length'].sum().reset_index()
    event_trips= df2.groupby(['agents','departure_time_sec','from_to','from_to_ser','hr'])['trip_time'].max().reset_index()
    event_trips = event_trips.merge(mat_length2, on=['agents','departure_time_sec','from_to','from_to_ser','hr'])
    event_trips.trip_time = event_trips.trip_time.astype(float).astype('int64')
    plan_event_trips = event_trips[event_trips.trip_time > 0]
    plan_event_trips.rename(columns={'trip_time':'MATSim Travel Time','link_length':'MATSim Distance'}, inplace=True)

    mat_gh_length_triptime = plan_event_trips.merge(gh_length_time, on=['agents','departure_time_sec'])  
    mat_gh_length_triptime.rename(columns={'GraphHopper Travel Time ori':'GraphHopper Travel Time'}, inplace=True)
    mat_gh_length_triptime[['agents','departure_time_sec','MATSim Travel Time','MATSim Distance','GraphHopper Travel Time','GraphHopper Distance']].to_csv('data/mat_gh_length_triptime.csv', index=False)

    ## stats -comparison
    avg_trip_len_mat_km = mat_gh_length_triptime['MATSim Distance'].sum() / len(mat_gh_length_triptime) / 1000
    avg_trip_len_mat_mile = avg_trip_len_mat_km * 0.6213711
    avg_trip_len_gh_km = mat_gh_length_triptime['GraphHopper Distance'].sum() / len(mat_gh_length_triptime) / 1000
    avg_trip_len_gh_mile = avg_trip_len_gh_km * 0.6213711
    avg_trip_time_mat_min = mat_gh_length_triptime['MATSim Travel Time'].sum() / len(mat_gh_length_triptime) /60
    avg_trip_time_gh_min = mat_gh_length_triptime['GraphHopper Travel Time'].sum() / len(mat_gh_length_triptime) /60
    len_percent_change = (avg_trip_len_mat_km - avg_trip_len_gh_km) / avg_trip_len_mat_km * 100
    time_percent_change = (avg_trip_time_mat_min - avg_trip_time_gh_min) / avg_trip_time_mat_min * 100

    avg_spent_time_mat_min = mat_gh_length_triptime['MATSim Travel Time'].sum() / len(mat_gh_length_triptime.agents.unique()) /60
    avg_spent_time_gh_min = mat_gh_length_triptime['GraphHopper Travel Time'].sum() / len(mat_gh_length_triptime.agents.unique().tolist()) /60
    spent_percent_change = (avg_spent_time_mat_min - avg_spent_time_gh_min) / avg_spent_time_mat_min * 100

    agents_total = mat_gh_length_triptime.groupby(['agents'])['GraphHopper Distance','GraphHopper Travel Time'].sum().reset_index()
    agents_total_mat = mat_gh_length_triptime.groupby(['agents'])['MATSim Distance','MATSim Travel Time'].sum().reset_index()

    vmt_mat_km = (agents_total_mat['MATSim Distance'].sum()/len(agents_total_mat))/1000
    vmt_mat_mi = vmt_mat_km * 0.6213711
    vmt_gh_km = (agents_total['GraphHopper Distance'].sum()/len(agents_total))/1000
    vmt_gh_mi = vmt_gh_km * 0.6213711
    vmt_percent_change_km = (vmt_mat_km-vmt_gh_km)/vmt_mat_km*100
    print( 'vmt_mat_km, vmt_mat_mi, vmt_gh_km, vmt_gh_mi, vmt_percent_change_km...............................')
    print( vmt_mat_km, vmt_mat_mi, vmt_gh_km, vmt_gh_mi, vmt_percent_change_km)
    print('avg_trip_len_mat_km, avg_trip_len_gh_km, len_percent_change..................')
    print(avg_trip_len_mat_km, avg_trip_len_gh_km, len_percent_change)
    print('avg_trip_time_mat_min, avg_trip_time_gh_min, time_percent_change..................')
    print(avg_trip_time_mat_min, avg_trip_time_gh_min, time_percent_change)
    print('avg_spent_time_mat_min, avg_spent_time_gh_min, spent_percent_change..................')
    print(avg_spent_time_mat_min, avg_spent_time_gh_min, spent_percent_change)

    return mat_gh_length_triptime, vmt_mat_km, vmt_mat_mi, vmt_gh_km, vmt_gh_mi, vmt_percent_change_km, avg_trip_len_mat_km, avg_trip_len_gh_km, len_percent_change, avg_trip_time_mat_min, avg_trip_time_gh_min, time_percent_change, avg_spent_time_mat_min, avg_spent_time_gh_min, spent_percent_change

def plan_fac():
    df_plan_faci = pd.read_csv('GraphHopper_DMV\\data\\dataforGH.csv')
    df_plan_faci['from_to'] = df_plan_faci['from_facility'] +'_'+ df_plan_faci['to_facility']
    df_plan_faci['from_to_ser'] = 1

    pop_data_frames = list(df_plan_faci.groupby('agents'))
    tot_agents=0
    for agent, agent_data in pop_data_frames:
        tot_agents+=1
        if tot_agents%50000 ==0:print(tot_agents)

        from_to_list= agent_data.from_to.tolist()
        from_to_list_unique= agent_data.from_to.unique().tolist()

        for l in from_to_list_unique:
            if from_to_list.count(l) > 1:
                idx_list= agent_data[agent_data.from_to == l].index.tolist()
                ser=0
                for idx in idx_list:
                    ser +=1
                    agent_data.loc[agent_data.index == idx, 'from_to_ser'] = ser

    df_plan_faci2 = pd.concat([df for _,df in pop_data_frames])
    return df_plan_faci2

def morning_traffic():
    mat_gh_length_triptime, a, b, c, d, e,f,g,h,i,j,k,l,m,n = mat_gh_length_time()
    morning_peak = mat_gh_length_triptime[(mat_gh_length_triptime['departure_time_sec'] >= 6*3600) & (mat_gh_length_triptime['departure_time_sec'] <= 9*3600)]
    morning_peak['mat_minus_gh_time_ori'] = morning_peak['MATSim Travel Time'] -  morning_peak['GraphHopper Travel Time']

    morning_peak['categoryAll'] =''
    morning_peak['categoryAll'] = np.where(morning_peak['mat_minus_gh_time_ori'] < 0, 'GraphHopper > MATSim', morning_peak['categoryAll'])
    morning_peak['categoryAll'] = np.where((morning_peak['mat_minus_gh_time_ori'] >= 0) & (morning_peak['mat_minus_gh_time_ori'] < 60), '1min', morning_peak['categoryAll'])
    morning_peak['categoryAll'] = np.where((morning_peak['mat_minus_gh_time_ori'] >= 60) & (morning_peak['mat_minus_gh_time_ori'] < 300), '5min', morning_peak['categoryAll'])
    morning_peak['categoryAll'] = np.where((morning_peak['mat_minus_gh_time_ori'] >= 300) & (morning_peak['mat_minus_gh_time_ori'] < 600), '10min', morning_peak['categoryAll'])
    morning_peak['categoryAll'] = np.where((morning_peak['mat_minus_gh_time_ori'] >= 600), '> 10min', morning_peak['categoryAll'])

    to_check = morning_peak[morning_peak.mat_minus_gh_time_ori < 0]
    to_check['category'] =''
    to_check['category'] = np.where(to_check['mat_minus_gh_time_ori'] < -300, '5min', to_check['category'])
    to_check['category'] = np.where((to_check['mat_minus_gh_time_ori'] >= -300) & (to_check['mat_minus_gh_time_ori'] < -240), '4min', to_check['category'])
    to_check['category'] = np.where((to_check['mat_minus_gh_time_ori'] >= -240) & (to_check['mat_minus_gh_time_ori'] < -180), '3min', to_check['category'])
    to_check['category'] = np.where((to_check['mat_minus_gh_time_ori'] >= -180) & (to_check['mat_minus_gh_time_ori'] < -120), '2min', to_check['category'])
    to_check['category'] = np.where((to_check['mat_minus_gh_time_ori'] >= -120) & (to_check['mat_minus_gh_time_ori'] < -60), '1min', to_check['category'])
    to_check['category'] = np.where((to_check['mat_minus_gh_time_ori'] >= -60) & (to_check['mat_minus_gh_time_ori'] < 0), '<1min', to_check['category'])

    x = 'GraphHopper Travel Time'
    y = 'MATSim Travel Time'

    morning_peak[['MATSim Travel Time','GraphHopper Travel Time','categoryAll']].to_csv('data/morning_peak.csv', index=False)
    to_check[['MATSim Travel Time','GraphHopper Travel Time','category']].to_csv('data/morning_peak_gh.csv', index=False)


def morning_plot_gh():
    morning_gh = pd.read_csv('data/morning_peak_gh.csv')

    color_dict = dict({'5min':'#ff0000',
                        '4min':'#82b74b',
                        '3min': '#BF40BF',
                        '2min': '#32BD68',
                        '1min': '#FFA500',
                        '<1min':'#00BFFF'})

    ax = sns.scatterplot(data=morning_gh, x=morning_gh['GraphHopper Travel Time'], y=morning_gh['MATSim Travel Time'], hue=morning_gh.category,  s=15, palette = color_dict) 
    ax.set_xlabel('GraphHopper Travel Time',fontsize=15, labelpad=15)
    ax.set_ylabel('MATSim Travel Time',fontsize=15, labelpad=15)
    plt.legend(fontsize=10, loc='lower right')
    ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('static/images/morning_traffic_gh.png')
    plt.close()
    # print('morning traffic plots saved.................')

def morning_plot_all():
    morning_peak = pd.read_csv('data/morning_peak.csv')

    color_dictAll = dict({'> 10min':'#ff0000',
                        '10min':'#82b74b',
                        '5min': '#BF40BF',
                        '1min': '#32BD68',
                        'GraphHopper > MATSim': '#FFA500'})

    ax = sns.scatterplot(data=morning_peak, x=morning_peak['GraphHopper Travel Time'], y=morning_peak['MATSim Travel Time'], hue=morning_peak.categoryAll,  s=15, palette = color_dictAll) 
    ax.set_xlabel('GraphHopper Travel Time',fontsize=15, labelpad=15)
    ax.set_ylabel('MATSim Travel Time',fontsize=15, labelpad=15)
    plt.legend(fontsize=10, loc='lower right')
    ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig('static/images/morning_traffic_all.png')
    plt.close()

def regression_dist():
    x = 'GraphHopper Distance'
    y = 'MATSim Distance'
    graphdata = pd.read_csv('data/mat_gh_length_triptime.csv')

    slope, intercept, r_value, p_value, std_err = stats.linregress(graphdata[x], graphdata[y])

    ax = sns.regplot(data=graphdata, x=x, y=y, color='lightblue', scatter_kws={'s':0.5}, line_kws={'color':'lightblue', 'label':"y={0:.4f}x+{1:.4f}".format(slope, intercept)})
    
    # Add scatter plot
    sns.scatterplot(data=graphdata, x=x, y=y, s=15, alpha=0.2, ax=ax)
    
    # Add text annotation
    ax.text(10000, 50000, "R2={0:.4f} p={1:.4f}".format(r_value, p_value), style='italic', fontsize=15)

    # Add identity line
    xpoints = (ax.get_xlim()[0], ax.get_ylim()[1])
    ypoints = (ax.get_xlim()[0], ax.get_ylim()[1])
    ax.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)

    # Add and specify legend
    plt.legend(fontsize=15, loc='lower right')
    
    # Set axis labels and ticks
    ax.set_xlabel('GraphHopper Distance (meter)', fontsize=15, labelpad=10)
    ax.set_ylabel('MATSim Distance (meter)', fontsize=15, labelpad=10)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('static/images/regression_dist.png')
    plt.close()


def regression_time():
    x = 'GraphHopper Travel Time'
    y = 'MATSim Travel Time'
    graphdata = pd.read_csv('data/mat_gh_length_triptime.csv')

    slope, intercept, r_value, p_value, std_err = stats.linregress(graphdata[x], graphdata[y])

    ax = sns.regplot(data=graphdata, x=x, y=y, color='lightblue', scatter_kws={'s':0.5}, 
                     line_kws={'color':'lightblue','label':"y={0:.4f}x+{1:.4f}".format(slope,intercept)})
    sns.scatterplot(data=graphdata, x=x, y=y, s=15, alpha=0.2, ax=ax)
    ax.text(200, 2500, "R2={0:.4f} p={1:.4f}".format(r_value,p_value), style ='italic', fontsize = 10)

    xpoints = (ax.get_xlim()[0], ax.get_ylim()[1])
    ypoints = (ax.get_xlim()[0], ax.get_ylim()[1])
    ax.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
    
    plt.legend(fontsize=10, loc='lower right')
    ax.set_xlabel('GraphHopper Travel Time (sec)',fontsize=10, labelpad=10)
    ax.set_ylabel('MATSim Travel Time (sec)',fontsize=10, labelpad=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig('static/images/regression_time.png')
    plt.close()
