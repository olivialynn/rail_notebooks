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

    <pzflow.flow.Flow at 0x7fe2edc7b880>



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
          <td>27.961073</td>
          <td>1.034387</td>
          <td>27.162497</td>
          <td>0.242338</td>
          <td>26.090388</td>
          <td>0.085389</td>
          <td>25.251067</td>
          <td>0.066332</td>
          <td>24.760667</td>
          <td>0.082202</td>
          <td>23.762879</td>
          <td>0.076698</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.398628</td>
          <td>1.323215</td>
          <td>27.242169</td>
          <td>0.258726</td>
          <td>26.748820</td>
          <td>0.151487</td>
          <td>26.344901</td>
          <td>0.172142</td>
          <td>26.445246</td>
          <td>0.341639</td>
          <td>25.821391</td>
          <td>0.429567</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.220423</td>
          <td>1.959534</td>
          <td>27.790551</td>
          <td>0.400165</td>
          <td>27.772596</td>
          <td>0.352820</td>
          <td>26.165849</td>
          <td>0.147712</td>
          <td>24.967072</td>
          <td>0.098560</td>
          <td>24.270934</td>
          <td>0.119757</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.895022</td>
          <td>0.994307</td>
          <td>31.496419</td>
          <td>2.905060</td>
          <td>28.504866</td>
          <td>0.609763</td>
          <td>26.390368</td>
          <td>0.178916</td>
          <td>25.656016</td>
          <td>0.178698</td>
          <td>25.410370</td>
          <td>0.311638</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.773348</td>
          <td>0.209613</td>
          <td>26.044346</td>
          <td>0.093159</td>
          <td>25.930712</td>
          <td>0.074165</td>
          <td>25.569880</td>
          <td>0.087910</td>
          <td>25.379762</td>
          <td>0.141109</td>
          <td>25.193578</td>
          <td>0.261504</td>
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
          <td>26.949112</td>
          <td>0.528823</td>
          <td>26.240864</td>
          <td>0.110628</td>
          <td>25.413343</td>
          <td>0.046873</td>
          <td>25.124079</td>
          <td>0.059268</td>
          <td>24.832887</td>
          <td>0.087602</td>
          <td>24.650084</td>
          <td>0.166005</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.232211</td>
          <td>0.256625</td>
          <td>25.917157</td>
          <td>0.073281</td>
          <td>25.208559</td>
          <td>0.063879</td>
          <td>24.683229</td>
          <td>0.076771</td>
          <td>24.230742</td>
          <td>0.115642</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.472314</td>
          <td>0.761030</td>
          <td>26.615345</td>
          <td>0.152920</td>
          <td>26.364855</td>
          <td>0.108636</td>
          <td>26.102598</td>
          <td>0.139886</td>
          <td>25.474798</td>
          <td>0.153117</td>
          <td>25.469822</td>
          <td>0.326768</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.001755</td>
          <td>0.253203</td>
          <td>26.188921</td>
          <td>0.105727</td>
          <td>26.112604</td>
          <td>0.087076</td>
          <td>25.799798</td>
          <td>0.107551</td>
          <td>25.555980</td>
          <td>0.164124</td>
          <td>24.807351</td>
          <td>0.189691</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.856029</td>
          <td>0.493910</td>
          <td>26.683493</td>
          <td>0.162094</td>
          <td>26.477968</td>
          <td>0.119887</td>
          <td>26.641651</td>
          <td>0.220977</td>
          <td>26.374765</td>
          <td>0.323071</td>
          <td>25.826189</td>
          <td>0.431136</td>
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
          <td>28.586327</td>
          <td>1.558796</td>
          <td>26.576180</td>
          <td>0.169934</td>
          <td>26.034111</td>
          <td>0.095543</td>
          <td>25.227001</td>
          <td>0.076957</td>
          <td>24.729792</td>
          <td>0.094051</td>
          <td>24.016714</td>
          <td>0.113262</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.958518</td>
          <td>1.002281</td>
          <td>26.652954</td>
          <td>0.163367</td>
          <td>25.936401</td>
          <td>0.143027</td>
          <td>25.857492</td>
          <td>0.246610</td>
          <td>25.513953</td>
          <td>0.392626</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.938917</td>
          <td>1.853059</td>
          <td>28.300010</td>
          <td>0.664927</td>
          <td>27.801924</td>
          <td>0.424337</td>
          <td>25.867500</td>
          <td>0.137839</td>
          <td>24.958652</td>
          <td>0.117452</td>
          <td>24.154219</td>
          <td>0.130548</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.968680</td>
          <td>1.161437</td>
          <td>29.700178</td>
          <td>1.558324</td>
          <td>27.207446</td>
          <td>0.276475</td>
          <td>26.352108</td>
          <td>0.217509</td>
          <td>26.027656</td>
          <td>0.301305</td>
          <td>25.421233</td>
          <td>0.388218</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.029270</td>
          <td>0.288725</td>
          <td>26.012147</td>
          <td>0.104491</td>
          <td>25.945661</td>
          <td>0.088429</td>
          <td>25.531459</td>
          <td>0.100623</td>
          <td>25.539584</td>
          <td>0.189215</td>
          <td>24.684352</td>
          <td>0.200757</td>
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
          <td>26.594925</td>
          <td>0.455260</td>
          <td>26.194775</td>
          <td>0.124771</td>
          <td>25.461443</td>
          <td>0.058855</td>
          <td>25.209168</td>
          <td>0.077430</td>
          <td>24.833944</td>
          <td>0.105215</td>
          <td>24.802421</td>
          <td>0.226073</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.930184</td>
          <td>0.575869</td>
          <td>26.783611</td>
          <td>0.203194</td>
          <td>26.029791</td>
          <td>0.095579</td>
          <td>25.143874</td>
          <td>0.071818</td>
          <td>24.782125</td>
          <td>0.098879</td>
          <td>24.223691</td>
          <td>0.136108</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.057595</td>
          <td>0.633373</td>
          <td>26.843248</td>
          <td>0.215167</td>
          <td>26.219551</td>
          <td>0.113801</td>
          <td>26.359030</td>
          <td>0.207405</td>
          <td>26.181851</td>
          <td>0.324409</td>
          <td>25.129903</td>
          <td>0.293275</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.729041</td>
          <td>0.506397</td>
          <td>26.197576</td>
          <td>0.126244</td>
          <td>26.136422</td>
          <td>0.107819</td>
          <td>25.976899</td>
          <td>0.152838</td>
          <td>25.661506</td>
          <td>0.215903</td>
          <td>25.161008</td>
          <td>0.306010</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.679614</td>
          <td>0.481548</td>
          <td>26.861178</td>
          <td>0.217892</td>
          <td>26.585687</td>
          <td>0.155723</td>
          <td>26.169696</td>
          <td>0.176329</td>
          <td>25.625244</td>
          <td>0.205243</td>
          <td>25.424969</td>
          <td>0.369705</td>
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
          <td>26.872817</td>
          <td>0.500107</td>
          <td>26.654803</td>
          <td>0.158188</td>
          <td>25.927622</td>
          <td>0.073972</td>
          <td>25.206443</td>
          <td>0.063769</td>
          <td>24.666593</td>
          <td>0.075661</td>
          <td>24.083808</td>
          <td>0.101732</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.151876</td>
          <td>0.611782</td>
          <td>27.206212</td>
          <td>0.251404</td>
          <td>26.603171</td>
          <td>0.133755</td>
          <td>26.135137</td>
          <td>0.144002</td>
          <td>25.743112</td>
          <td>0.192525</td>
          <td>25.250521</td>
          <td>0.274184</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.546023</td>
          <td>2.124431</td>
          <td>28.181607</td>
          <td>0.516787</td>
          <td>26.037165</td>
          <td>0.143786</td>
          <td>24.817958</td>
          <td>0.093838</td>
          <td>24.267905</td>
          <td>0.129870</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.535805</td>
          <td>0.801280</td>
          <td>27.310671</td>
          <td>0.299545</td>
          <td>26.068419</td>
          <td>0.170677</td>
          <td>25.887021</td>
          <td>0.268004</td>
          <td>24.809573</td>
          <td>0.236970</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.117359</td>
          <td>0.278470</td>
          <td>25.978402</td>
          <td>0.088027</td>
          <td>25.912446</td>
          <td>0.073081</td>
          <td>25.801646</td>
          <td>0.107885</td>
          <td>25.156456</td>
          <td>0.116457</td>
          <td>24.957551</td>
          <td>0.215473</td>
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
          <td>26.355640</td>
          <td>0.353987</td>
          <td>26.326618</td>
          <td>0.127603</td>
          <td>25.484204</td>
          <td>0.054062</td>
          <td>25.102314</td>
          <td>0.063185</td>
          <td>25.020230</td>
          <td>0.111660</td>
          <td>24.980417</td>
          <td>0.236868</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.833718</td>
          <td>0.490543</td>
          <td>26.738299</td>
          <td>0.172212</td>
          <td>26.031946</td>
          <td>0.082455</td>
          <td>25.308498</td>
          <td>0.071021</td>
          <td>24.842924</td>
          <td>0.089849</td>
          <td>24.024017</td>
          <td>0.098190</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.628117</td>
          <td>0.863577</td>
          <td>26.914079</td>
          <td>0.205265</td>
          <td>26.543898</td>
          <td>0.133217</td>
          <td>26.081978</td>
          <td>0.144503</td>
          <td>25.799474</td>
          <td>0.211302</td>
          <td>25.697754</td>
          <td>0.408471</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.631735</td>
          <td>0.895702</td>
          <td>26.337298</td>
          <td>0.132901</td>
          <td>26.104933</td>
          <td>0.097012</td>
          <td>25.807742</td>
          <td>0.121960</td>
          <td>25.657641</td>
          <td>0.199751</td>
          <td>25.424446</td>
          <td>0.350760</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.846828</td>
          <td>0.501928</td>
          <td>26.697553</td>
          <td>0.169533</td>
          <td>26.464507</td>
          <td>0.123143</td>
          <td>26.234538</td>
          <td>0.163016</td>
          <td>26.240314</td>
          <td>0.300570</td>
          <td>25.413952</td>
          <td>0.324197</td>
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
