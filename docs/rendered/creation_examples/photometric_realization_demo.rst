Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fc58b0b42b0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>27.823946</td>
          <td>0.952256</td>
          <td>26.567645</td>
          <td>0.146790</td>
          <td>26.256230</td>
          <td>0.098787</td>
          <td>25.111856</td>
          <td>0.058628</td>
          <td>24.673644</td>
          <td>0.076124</td>
          <td>24.012753</td>
          <td>0.095576</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.751752</td>
          <td>1.583295</td>
          <td>27.168825</td>
          <td>0.243605</td>
          <td>26.859105</td>
          <td>0.166467</td>
          <td>26.329032</td>
          <td>0.169834</td>
          <td>25.552217</td>
          <td>0.163598</td>
          <td>25.464637</td>
          <td>0.325424</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.335470</td>
          <td>0.598833</td>
          <td>27.998120</td>
          <td>0.420186</td>
          <td>26.043901</td>
          <td>0.132974</td>
          <td>24.878898</td>
          <td>0.091220</td>
          <td>24.404307</td>
          <td>0.134431</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.167489</td>
          <td>0.530797</td>
          <td>27.799935</td>
          <td>0.360469</td>
          <td>26.268683</td>
          <td>0.161316</td>
          <td>25.206987</td>
          <td>0.121517</td>
          <td>25.606580</td>
          <td>0.363974</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.880730</td>
          <td>0.229183</td>
          <td>26.101114</td>
          <td>0.097912</td>
          <td>26.043461</td>
          <td>0.081929</td>
          <td>25.689260</td>
          <td>0.097632</td>
          <td>25.604664</td>
          <td>0.171074</td>
          <td>24.993885</td>
          <td>0.221785</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.377700</td>
          <td>0.124600</td>
          <td>25.561793</td>
          <td>0.053477</td>
          <td>25.101806</td>
          <td>0.058108</td>
          <td>25.006861</td>
          <td>0.102056</td>
          <td>24.798883</td>
          <td>0.188341</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.214504</td>
          <td>0.300894</td>
          <td>26.918992</td>
          <td>0.197874</td>
          <td>26.115536</td>
          <td>0.087301</td>
          <td>25.104456</td>
          <td>0.058245</td>
          <td>24.918227</td>
          <td>0.094426</td>
          <td>24.264129</td>
          <td>0.119051</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>31.818683</td>
          <td>4.381881</td>
          <td>26.598987</td>
          <td>0.150791</td>
          <td>26.310571</td>
          <td>0.103602</td>
          <td>26.220753</td>
          <td>0.154836</td>
          <td>25.912766</td>
          <td>0.221716</td>
          <td>25.911625</td>
          <td>0.459869</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.307452</td>
          <td>0.324082</td>
          <td>26.186726</td>
          <td>0.105524</td>
          <td>26.150907</td>
          <td>0.090061</td>
          <td>25.749976</td>
          <td>0.102966</td>
          <td>26.248673</td>
          <td>0.292015</td>
          <td>25.487535</td>
          <td>0.331396</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.539910</td>
          <td>0.388869</td>
          <td>26.671035</td>
          <td>0.160380</td>
          <td>26.569237</td>
          <td>0.129765</td>
          <td>26.411896</td>
          <td>0.182209</td>
          <td>25.917427</td>
          <td>0.222578</td>
          <td>26.563017</td>
          <td>0.731039</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.415578</td>
          <td>0.148151</td>
          <td>26.011607</td>
          <td>0.093674</td>
          <td>25.231643</td>
          <td>0.077273</td>
          <td>24.722230</td>
          <td>0.093429</td>
          <td>24.076291</td>
          <td>0.119289</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.328146</td>
          <td>0.316369</td>
          <td>26.567490</td>
          <td>0.151851</td>
          <td>26.268767</td>
          <td>0.189875</td>
          <td>25.847660</td>
          <td>0.244622</td>
          <td>25.246104</td>
          <td>0.318147</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.856161</td>
          <td>1.642854</td>
          <td>27.985370</td>
          <td>0.487105</td>
          <td>25.900924</td>
          <td>0.141867</td>
          <td>25.033145</td>
          <td>0.125300</td>
          <td>24.209989</td>
          <td>0.136991</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>31.163153</td>
          <td>3.928756</td>
          <td>28.241259</td>
          <td>0.659265</td>
          <td>27.721198</td>
          <td>0.414815</td>
          <td>26.374550</td>
          <td>0.221611</td>
          <td>25.445952</td>
          <td>0.186418</td>
          <td>25.325843</td>
          <td>0.360432</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.204911</td>
          <td>0.332242</td>
          <td>26.146143</td>
          <td>0.117427</td>
          <td>26.065913</td>
          <td>0.098277</td>
          <td>25.724729</td>
          <td>0.119108</td>
          <td>25.461949</td>
          <td>0.177188</td>
          <td>24.921597</td>
          <td>0.244553</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.718513</td>
          <td>0.499127</td>
          <td>26.435069</td>
          <td>0.153462</td>
          <td>25.482643</td>
          <td>0.059971</td>
          <td>25.024963</td>
          <td>0.065791</td>
          <td>24.758691</td>
          <td>0.098509</td>
          <td>24.552079</td>
          <td>0.183277</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.522519</td>
          <td>0.859344</td>
          <td>26.637518</td>
          <td>0.179663</td>
          <td>25.911212</td>
          <td>0.086120</td>
          <td>25.156898</td>
          <td>0.072650</td>
          <td>24.771947</td>
          <td>0.098001</td>
          <td>24.431041</td>
          <td>0.162623</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.095385</td>
          <td>0.307135</td>
          <td>26.714762</td>
          <td>0.193209</td>
          <td>26.597928</td>
          <td>0.157784</td>
          <td>26.424007</td>
          <td>0.218969</td>
          <td>26.095567</td>
          <td>0.302790</td>
          <td>25.158475</td>
          <td>0.300100</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.270593</td>
          <td>0.357422</td>
          <td>26.125843</td>
          <td>0.118632</td>
          <td>26.154823</td>
          <td>0.109565</td>
          <td>25.922938</td>
          <td>0.145920</td>
          <td>25.660943</td>
          <td>0.215801</td>
          <td>25.071964</td>
          <td>0.284828</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.224145</td>
          <td>0.339584</td>
          <td>26.527597</td>
          <td>0.164488</td>
          <td>26.717823</td>
          <td>0.174295</td>
          <td>26.300058</td>
          <td>0.196857</td>
          <td>26.093463</td>
          <td>0.301522</td>
          <td>25.330671</td>
          <td>0.343341</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398945</td>
          <td>29.660073</td>
          <td>2.338069</td>
          <td>26.610660</td>
          <td>0.152324</td>
          <td>26.050960</td>
          <td>0.082484</td>
          <td>25.259057</td>
          <td>0.066813</td>
          <td>24.721703</td>
          <td>0.079435</td>
          <td>24.062485</td>
          <td>0.099850</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.976233</td>
          <td>0.539641</td>
          <td>27.544167</td>
          <td>0.330297</td>
          <td>26.659758</td>
          <td>0.140449</td>
          <td>26.025740</td>
          <td>0.131030</td>
          <td>27.000409</td>
          <td>0.521829</td>
          <td>25.043719</td>
          <td>0.231369</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.526418</td>
          <td>0.660488</td>
          <td>25.965526</td>
          <td>0.135175</td>
          <td>25.066755</td>
          <td>0.116639</td>
          <td>24.194234</td>
          <td>0.121835</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.452336</td>
          <td>0.849517</td>
          <td>28.478229</td>
          <td>0.771621</td>
          <td>27.113992</td>
          <td>0.255326</td>
          <td>26.407980</td>
          <td>0.227061</td>
          <td>25.549312</td>
          <td>0.202669</td>
          <td>24.826194</td>
          <td>0.240244</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.863109</td>
          <td>0.226072</td>
          <td>26.148437</td>
          <td>0.102177</td>
          <td>25.931792</td>
          <td>0.074342</td>
          <td>25.594105</td>
          <td>0.089939</td>
          <td>25.620652</td>
          <td>0.173653</td>
          <td>25.225762</td>
          <td>0.268835</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.716841</td>
          <td>0.927021</td>
          <td>26.263169</td>
          <td>0.120776</td>
          <td>25.425944</td>
          <td>0.051337</td>
          <td>25.101169</td>
          <td>0.063121</td>
          <td>24.819284</td>
          <td>0.093654</td>
          <td>24.838589</td>
          <td>0.210524</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.372243</td>
          <td>2.098926</td>
          <td>26.702617</td>
          <td>0.167064</td>
          <td>26.169587</td>
          <td>0.093074</td>
          <td>25.281845</td>
          <td>0.069365</td>
          <td>24.878298</td>
          <td>0.092686</td>
          <td>24.107608</td>
          <td>0.105644</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.728842</td>
          <td>0.462609</td>
          <td>26.825359</td>
          <td>0.190520</td>
          <td>26.354951</td>
          <td>0.113064</td>
          <td>25.983062</td>
          <td>0.132686</td>
          <td>25.881785</td>
          <td>0.226299</td>
          <td>25.644484</td>
          <td>0.392057</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.015855</td>
          <td>0.593365</td>
          <td>26.036559</td>
          <td>0.102334</td>
          <td>26.101017</td>
          <td>0.096679</td>
          <td>25.929501</td>
          <td>0.135521</td>
          <td>25.459961</td>
          <td>0.168999</td>
          <td>25.074847</td>
          <td>0.265006</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.643454</td>
          <td>0.867516</td>
          <td>26.831248</td>
          <td>0.189857</td>
          <td>26.456083</td>
          <td>0.122245</td>
          <td>26.538080</td>
          <td>0.210687</td>
          <td>26.260992</td>
          <td>0.305603</td>
          <td>25.728094</td>
          <td>0.414324</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
