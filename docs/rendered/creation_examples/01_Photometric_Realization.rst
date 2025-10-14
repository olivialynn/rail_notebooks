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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fd40e675390>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.144472  0.087725  
    1      25.391064  0.234121  0.131014  
    2      24.304707  0.054692  0.028487  
    3      25.291103  0.092160  0.076648  
    4      25.096743  0.036220  0.033710  
    ...          ...       ...       ...  
    99995  24.737946  0.126219  0.117872  
    99996  24.224169  0.205964  0.186940  
    99997  25.613836  0.117924  0.092071  
    99998  25.274899  0.022737  0.011429  
    99999  25.699642  0.195500  0.113796  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.605417</td>
          <td>0.151625</td>
          <td>26.104213</td>
          <td>0.086435</td>
          <td>25.212519</td>
          <td>0.064104</td>
          <td>24.631902</td>
          <td>0.073367</td>
          <td>24.024499</td>
          <td>0.096566</td>
          <td>0.144472</td>
          <td>0.087725</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.069134</td>
          <td>0.493833</td>
          <td>26.652750</td>
          <td>0.139474</td>
          <td>26.363924</td>
          <td>0.174947</td>
          <td>25.500437</td>
          <td>0.156517</td>
          <td>25.052651</td>
          <td>0.232871</td>
          <td>0.234121</td>
          <td>0.131014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.767318</td>
          <td>0.803090</td>
          <td>27.841297</td>
          <td>0.372308</td>
          <td>26.010055</td>
          <td>0.129137</td>
          <td>24.907679</td>
          <td>0.093556</td>
          <td>24.417165</td>
          <td>0.135932</td>
          <td>0.054692</td>
          <td>0.028487</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.671685</td>
          <td>2.348248</td>
          <td>30.136953</td>
          <td>1.717960</td>
          <td>27.360985</td>
          <td>0.253423</td>
          <td>26.015769</td>
          <td>0.129777</td>
          <td>25.513152</td>
          <td>0.158229</td>
          <td>25.081603</td>
          <td>0.238513</td>
          <td>0.092160</td>
          <td>0.076648</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.323842</td>
          <td>0.328326</td>
          <td>26.151508</td>
          <td>0.102326</td>
          <td>26.006169</td>
          <td>0.079277</td>
          <td>25.625640</td>
          <td>0.092329</td>
          <td>25.253425</td>
          <td>0.126514</td>
          <td>25.208791</td>
          <td>0.264775</td>
          <td>0.036220</td>
          <td>0.033710</td>
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
          <td>27.501862</td>
          <td>0.776009</td>
          <td>26.465888</td>
          <td>0.134476</td>
          <td>25.426396</td>
          <td>0.047420</td>
          <td>25.067226</td>
          <td>0.056351</td>
          <td>24.826779</td>
          <td>0.087132</td>
          <td>24.829197</td>
          <td>0.193217</td>
          <td>0.126219</td>
          <td>0.117872</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.353474</td>
          <td>0.702814</td>
          <td>26.903904</td>
          <td>0.195380</td>
          <td>26.234562</td>
          <td>0.096927</td>
          <td>25.238152</td>
          <td>0.065577</td>
          <td>24.865790</td>
          <td>0.090174</td>
          <td>24.350137</td>
          <td>0.128277</td>
          <td>0.205964</td>
          <td>0.186940</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.134866</td>
          <td>1.144359</td>
          <td>26.555591</td>
          <td>0.145278</td>
          <td>26.468472</td>
          <td>0.118902</td>
          <td>26.187967</td>
          <td>0.150544</td>
          <td>25.877648</td>
          <td>0.215323</td>
          <td>27.077569</td>
          <td>1.013388</td>
          <td>0.117924</td>
          <td>0.092071</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.082426</td>
          <td>0.582125</td>
          <td>26.171711</td>
          <td>0.104149</td>
          <td>26.050655</td>
          <td>0.082451</td>
          <td>25.803662</td>
          <td>0.107915</td>
          <td>25.643433</td>
          <td>0.176802</td>
          <td>25.692505</td>
          <td>0.389133</td>
          <td>0.022737</td>
          <td>0.011429</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.931727</td>
          <td>0.522160</td>
          <td>26.811966</td>
          <td>0.180796</td>
          <td>26.554880</td>
          <td>0.128162</td>
          <td>26.451017</td>
          <td>0.188336</td>
          <td>26.389263</td>
          <td>0.326818</td>
          <td>25.864606</td>
          <td>0.443870</td>
          <td>0.195500</td>
          <td>0.113796</td>
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
          <td>1.398944</td>
          <td>26.827335</td>
          <td>0.549107</td>
          <td>26.701578</td>
          <td>0.196658</td>
          <td>25.910532</td>
          <td>0.089747</td>
          <td>25.077053</td>
          <td>0.070707</td>
          <td>24.775657</td>
          <td>0.102497</td>
          <td>23.940680</td>
          <td>0.111076</td>
          <td>0.144472</td>
          <td>0.087725</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.820942</td>
          <td>0.502953</td>
          <td>26.716079</td>
          <td>0.190968</td>
          <td>26.045134</td>
          <td>0.174616</td>
          <td>25.853130</td>
          <td>0.271263</td>
          <td>25.127079</td>
          <td>0.319356</td>
          <td>0.234121</td>
          <td>0.131014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.156146</td>
          <td>1.870049</td>
          <td>27.990870</td>
          <td>0.482572</td>
          <td>25.985821</td>
          <td>0.150170</td>
          <td>24.918976</td>
          <td>0.111688</td>
          <td>24.092695</td>
          <td>0.121790</td>
          <td>0.054692</td>
          <td>0.028487</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.902294</td>
          <td>0.571447</td>
          <td>28.135224</td>
          <td>0.593242</td>
          <td>27.419139</td>
          <td>0.315125</td>
          <td>26.491365</td>
          <td>0.234075</td>
          <td>25.503432</td>
          <td>0.187717</td>
          <td>25.035434</td>
          <td>0.274534</td>
          <td>0.092160</td>
          <td>0.076648</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.536993</td>
          <td>0.430972</td>
          <td>26.058799</td>
          <td>0.109203</td>
          <td>25.968473</td>
          <td>0.090565</td>
          <td>25.617013</td>
          <td>0.108864</td>
          <td>25.246469</td>
          <td>0.147963</td>
          <td>24.638237</td>
          <td>0.193846</td>
          <td>0.036220</td>
          <td>0.033710</td>
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
          <td>26.056326</td>
          <td>0.305194</td>
          <td>26.177914</td>
          <td>0.126003</td>
          <td>25.392539</td>
          <td>0.056916</td>
          <td>25.155068</td>
          <td>0.075942</td>
          <td>24.736981</td>
          <td>0.099319</td>
          <td>24.404893</td>
          <td>0.166181</td>
          <td>0.126219</td>
          <td>0.117872</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.627179</td>
          <td>0.196488</td>
          <td>25.908606</td>
          <td>0.096137</td>
          <td>25.159199</td>
          <td>0.081786</td>
          <td>24.759153</td>
          <td>0.108362</td>
          <td>24.093697</td>
          <td>0.136210</td>
          <td>0.205964</td>
          <td>0.186940</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.213755</td>
          <td>0.715096</td>
          <td>26.750071</td>
          <td>0.203173</td>
          <td>26.513234</td>
          <td>0.150190</td>
          <td>26.180226</td>
          <td>0.182682</td>
          <td>25.678203</td>
          <td>0.220005</td>
          <td>25.102234</td>
          <td>0.293308</td>
          <td>0.117924</td>
          <td>0.092071</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.113841</td>
          <td>0.653966</td>
          <td>26.102161</td>
          <td>0.113096</td>
          <td>26.090111</td>
          <td>0.100457</td>
          <td>25.939444</td>
          <td>0.143527</td>
          <td>25.605815</td>
          <td>0.200206</td>
          <td>25.158896</td>
          <td>0.296912</td>
          <td>0.022737</td>
          <td>0.011429</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.433922</td>
          <td>0.418986</td>
          <td>26.556354</td>
          <td>0.179067</td>
          <td>26.486444</td>
          <td>0.152937</td>
          <td>26.224977</td>
          <td>0.197773</td>
          <td>25.850913</td>
          <td>0.263899</td>
          <td>25.105527</td>
          <td>0.305911</td>
          <td>0.195500</td>
          <td>0.113796</td>
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
          <td>1.398944</td>
          <td>28.842709</td>
          <td>1.758114</td>
          <td>26.627272</td>
          <td>0.177087</td>
          <td>25.986574</td>
          <td>0.091373</td>
          <td>25.215113</td>
          <td>0.075912</td>
          <td>24.766787</td>
          <td>0.096874</td>
          <td>24.052805</td>
          <td>0.116520</td>
          <td>0.144472</td>
          <td>0.087725</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.309445</td>
          <td>0.355352</td>
          <td>26.957648</td>
          <td>0.245262</td>
          <td>26.427165</td>
          <td>0.252585</td>
          <td>25.814398</td>
          <td>0.275682</td>
          <td>25.037839</td>
          <td>0.311866</td>
          <td>0.234121</td>
          <td>0.131014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.569840</td>
          <td>0.821719</td>
          <td>28.937162</td>
          <td>0.909091</td>
          <td>28.051076</td>
          <td>0.446949</td>
          <td>25.947430</td>
          <td>0.125516</td>
          <td>25.019093</td>
          <td>0.105749</td>
          <td>24.092935</td>
          <td>0.105217</td>
          <td>0.054692</td>
          <td>0.028487</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.095248</td>
          <td>0.617996</td>
          <td>28.186257</td>
          <td>0.573484</td>
          <td>27.575794</td>
          <td>0.327046</td>
          <td>26.664781</td>
          <td>0.245798</td>
          <td>25.935662</td>
          <td>0.245608</td>
          <td>25.463263</td>
          <td>0.353048</td>
          <td>0.092160</td>
          <td>0.076648</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.869305</td>
          <td>0.229476</td>
          <td>26.193192</td>
          <td>0.107633</td>
          <td>26.030409</td>
          <td>0.082334</td>
          <td>25.547438</td>
          <td>0.087688</td>
          <td>25.619259</td>
          <td>0.175966</td>
          <td>25.001860</td>
          <td>0.226876</td>
          <td>0.036220</td>
          <td>0.033710</td>
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
          <td>28.516244</td>
          <td>1.510590</td>
          <td>26.533884</td>
          <td>0.164982</td>
          <td>25.462529</td>
          <td>0.058102</td>
          <td>25.146602</td>
          <td>0.072230</td>
          <td>24.823569</td>
          <td>0.102856</td>
          <td>24.756653</td>
          <td>0.214749</td>
          <td>0.126219</td>
          <td>0.117872</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.092677</td>
          <td>1.323998</td>
          <td>27.193750</td>
          <td>0.333497</td>
          <td>26.094982</td>
          <td>0.122010</td>
          <td>25.208495</td>
          <td>0.092366</td>
          <td>24.719382</td>
          <td>0.112888</td>
          <td>24.121647</td>
          <td>0.150554</td>
          <td>0.205964</td>
          <td>0.186940</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.966935</td>
          <td>0.578905</td>
          <td>26.450769</td>
          <td>0.148695</td>
          <td>26.320319</td>
          <td>0.119038</td>
          <td>26.281367</td>
          <td>0.186159</td>
          <td>25.777624</td>
          <td>0.224323</td>
          <td>25.157171</td>
          <td>0.287843</td>
          <td>0.117924</td>
          <td>0.092071</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.141046</td>
          <td>0.284397</td>
          <td>26.020929</td>
          <td>0.091612</td>
          <td>26.223223</td>
          <td>0.096388</td>
          <td>26.074922</td>
          <td>0.137204</td>
          <td>25.911901</td>
          <td>0.222475</td>
          <td>25.190880</td>
          <td>0.262034</td>
          <td>0.022737</td>
          <td>0.011429</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.644663</td>
          <td>0.494957</td>
          <td>26.902439</td>
          <td>0.241506</td>
          <td>26.681057</td>
          <td>0.182355</td>
          <td>26.498172</td>
          <td>0.250682</td>
          <td>25.673391</td>
          <td>0.230288</td>
          <td>24.884918</td>
          <td>0.258324</td>
          <td>0.195500</td>
          <td>0.113796</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
