# AdpGen Synthetic Workload Generator (2018 version)
AdpGen, a workload generator for live events broadcasted through HTTP Adaptive Streaming. It can be used, for instance, to generate a synthetic workload in order to evaluate server resource consumption.

<ul>
<li>Based on 2018 FIFA World Cup event: https://link.springer.com/article/10.1007/s00530-021-00788-4</li>
<li>Generates the client on-time, inter-session time, number of sessions and fraction of requested segments</li>
</ul>
  
<h1>Instructions</h1>
<p>To call the generator from the command line use:<br /> 
<i>python3 generator_cluster_markov_thesis_LITE.py [execution round number] [random seed];</i></p>
<p>
Example: python3 generator_cluster_markov_thesis_LITE.py 1 1276;<br />
</p>
<p>
The output is stored in the <b>synthetic</b> folder:
<ul>
<li>cluster_ontime_x.csv: session duration in seconds</li>
<li>cluster_offtime_x.csv: client inter-session duration</li>
<li>cluster_qtd_x.csv: client number of sessions</li>
<li>cluster_other_x.csv: performance metrics (average bitrate, number of bytes, number of segments of each bitrate)</li>
</ul>
</p>
  
<h1>Number of clients and arrival regime</h1>

<p>
<b>Number of clients</b><br/>
To change the number of clients, change the value of variable <i>TOTAL_CLIENTS</i> in line 150
</p>
