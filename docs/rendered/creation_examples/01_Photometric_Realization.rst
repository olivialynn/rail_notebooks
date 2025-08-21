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

    <pzflow.flow.Flow at 0x7f556004aef0>



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
    0      23.994413  0.101151  0.082698  
    1      25.391064  0.059318  0.041850  
    2      24.304707  0.119045  0.118783  
    3      25.291103  0.008205  0.004428  
    4      25.096743  0.014849  0.010345  
    ...          ...       ...       ...  
    99995  24.737946  0.021984  0.018526  
    99996  24.224169  0.045849  0.041437  
    99997  25.613836  0.027045  0.016842  
    99998  25.274899  0.107540  0.099766  
    99999  25.699642  0.037790  0.025562  
    
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
          <td>26.748697</td>
          <td>0.171349</td>
          <td>26.018010</td>
          <td>0.080110</td>
          <td>25.152332</td>
          <td>0.060772</td>
          <td>24.694822</td>
          <td>0.077562</td>
          <td>24.032197</td>
          <td>0.097220</td>
          <td>0.101151</td>
          <td>0.082698</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.517545</td>
          <td>0.323144</td>
          <td>26.633648</td>
          <td>0.137195</td>
          <td>26.128059</td>
          <td>0.142988</td>
          <td>26.011110</td>
          <td>0.240541</td>
          <td>25.129745</td>
          <td>0.248167</td>
          <td>0.059318</td>
          <td>0.041850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.157660</td>
          <td>1.024393</td>
          <td>27.907916</td>
          <td>0.392061</td>
          <td>26.112042</td>
          <td>0.141029</td>
          <td>25.115056</td>
          <td>0.112174</td>
          <td>24.142527</td>
          <td>0.107078</td>
          <td>0.119045</td>
          <td>0.118783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.092134</td>
          <td>2.534002</td>
          <td>27.159039</td>
          <td>0.214409</td>
          <td>26.620555</td>
          <td>0.217128</td>
          <td>25.735888</td>
          <td>0.191183</td>
          <td>25.787435</td>
          <td>0.418593</td>
          <td>0.008205</td>
          <td>0.004428</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.723482</td>
          <td>0.447377</td>
          <td>26.043863</td>
          <td>0.093120</td>
          <td>26.063920</td>
          <td>0.083420</td>
          <td>25.822835</td>
          <td>0.109736</td>
          <td>25.558131</td>
          <td>0.164426</td>
          <td>24.798433</td>
          <td>0.188269</td>
          <td>0.014849</td>
          <td>0.010345</td>
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
          <td>27.914187</td>
          <td>1.005838</td>
          <td>26.425889</td>
          <td>0.129908</td>
          <td>25.483368</td>
          <td>0.049880</td>
          <td>25.032361</td>
          <td>0.054634</td>
          <td>24.939685</td>
          <td>0.096221</td>
          <td>24.922312</td>
          <td>0.208928</td>
          <td>0.021984</td>
          <td>0.018526</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.070105</td>
          <td>0.577034</td>
          <td>26.845570</td>
          <td>0.186006</td>
          <td>25.871333</td>
          <td>0.070370</td>
          <td>25.351617</td>
          <td>0.072508</td>
          <td>24.820337</td>
          <td>0.086639</td>
          <td>24.525924</td>
          <td>0.149276</td>
          <td>0.045849</td>
          <td>0.041437</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.778253</td>
          <td>0.466159</td>
          <td>26.903051</td>
          <td>0.195240</td>
          <td>26.234181</td>
          <td>0.096895</td>
          <td>26.386271</td>
          <td>0.178296</td>
          <td>26.112039</td>
          <td>0.261336</td>
          <td>25.140630</td>
          <td>0.250398</td>
          <td>0.027045</td>
          <td>0.016842</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.045538</td>
          <td>0.262432</td>
          <td>26.181531</td>
          <td>0.105046</td>
          <td>26.156331</td>
          <td>0.090491</td>
          <td>25.788078</td>
          <td>0.106455</td>
          <td>25.824163</td>
          <td>0.205907</td>
          <td>25.531895</td>
          <td>0.343233</td>
          <td>0.107540</td>
          <td>0.099766</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.697089</td>
          <td>0.438547</td>
          <td>26.589127</td>
          <td>0.149522</td>
          <td>26.517018</td>
          <td>0.124023</td>
          <td>26.205248</td>
          <td>0.152792</td>
          <td>26.142893</td>
          <td>0.268004</td>
          <td>25.874698</td>
          <td>0.447266</td>
          <td>0.037790</td>
          <td>0.025562</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.954760</td>
          <td>0.239146</td>
          <td>25.997146</td>
          <td>0.095137</td>
          <td>25.204039</td>
          <td>0.077658</td>
          <td>24.682686</td>
          <td>0.092820</td>
          <td>23.974012</td>
          <td>0.112306</td>
          <td>0.101151</td>
          <td>0.082698</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.148210</td>
          <td>0.672744</td>
          <td>27.325719</td>
          <td>0.318003</td>
          <td>26.732197</td>
          <td>0.176234</td>
          <td>26.093611</td>
          <td>0.165084</td>
          <td>25.823139</td>
          <td>0.241679</td>
          <td>24.878826</td>
          <td>0.238027</td>
          <td>0.059318</td>
          <td>0.041850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.274713</td>
          <td>2.152888</td>
          <td>31.699027</td>
          <td>3.293455</td>
          <td>28.535555</td>
          <td>0.732365</td>
          <td>26.110822</td>
          <td>0.173847</td>
          <td>24.820438</td>
          <td>0.106588</td>
          <td>24.202592</td>
          <td>0.139396</td>
          <td>0.119045</td>
          <td>0.118783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.873141</td>
          <td>0.551380</td>
          <td>28.351445</td>
          <td>0.678127</td>
          <td>27.053572</td>
          <td>0.228876</td>
          <td>26.779647</td>
          <td>0.289621</td>
          <td>25.512883</td>
          <td>0.184959</td>
          <td>25.598960</td>
          <td>0.419082</td>
          <td>0.008205</td>
          <td>0.004428</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.720766</td>
          <td>0.493437</td>
          <td>26.024626</td>
          <td>0.105654</td>
          <td>26.021569</td>
          <td>0.094547</td>
          <td>26.040427</td>
          <td>0.156434</td>
          <td>25.468581</td>
          <td>0.178221</td>
          <td>24.872941</td>
          <td>0.234972</td>
          <td>0.014849</td>
          <td>0.010345</td>
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
          <td>26.403596</td>
          <td>0.146813</td>
          <td>25.415007</td>
          <td>0.055369</td>
          <td>25.152556</td>
          <td>0.072161</td>
          <td>24.823668</td>
          <td>0.102256</td>
          <td>24.899665</td>
          <td>0.240414</td>
          <td>0.021984</td>
          <td>0.018526</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.748424</td>
          <td>0.505504</td>
          <td>26.578433</td>
          <td>0.171219</td>
          <td>26.051508</td>
          <td>0.097637</td>
          <td>25.105096</td>
          <td>0.069561</td>
          <td>24.654895</td>
          <td>0.088630</td>
          <td>24.117914</td>
          <td>0.124489</td>
          <td>0.045849</td>
          <td>0.041437</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.048988</td>
          <td>0.293630</td>
          <td>26.626734</td>
          <td>0.177646</td>
          <td>26.377013</td>
          <td>0.129054</td>
          <td>26.307602</td>
          <td>0.196479</td>
          <td>25.926642</td>
          <td>0.261365</td>
          <td>25.659649</td>
          <td>0.439490</td>
          <td>0.027045</td>
          <td>0.016842</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.868243</td>
          <td>0.259704</td>
          <td>26.178046</td>
          <td>0.124560</td>
          <td>26.061410</td>
          <td>0.101367</td>
          <td>25.664973</td>
          <td>0.117201</td>
          <td>25.453391</td>
          <td>0.181937</td>
          <td>25.227634</td>
          <td>0.323908</td>
          <td>0.107540</td>
          <td>0.099766</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.053664</td>
          <td>0.628097</td>
          <td>26.834580</td>
          <td>0.211922</td>
          <td>26.533613</td>
          <td>0.147976</td>
          <td>26.032871</td>
          <td>0.155893</td>
          <td>26.002166</td>
          <td>0.278416</td>
          <td>25.385368</td>
          <td>0.356292</td>
          <td>0.037790</td>
          <td>0.025562</td>
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
          <td>26.964528</td>
          <td>0.568281</td>
          <td>26.578751</td>
          <td>0.161939</td>
          <td>26.087677</td>
          <td>0.094495</td>
          <td>25.126366</td>
          <td>0.066253</td>
          <td>24.749727</td>
          <td>0.090315</td>
          <td>23.975876</td>
          <td>0.102987</td>
          <td>0.101151</td>
          <td>0.082698</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.208736</td>
          <td>0.258871</td>
          <td>26.796574</td>
          <td>0.163183</td>
          <td>26.713235</td>
          <td>0.242589</td>
          <td>25.946388</td>
          <td>0.235505</td>
          <td>25.108463</td>
          <td>0.252146</td>
          <td>0.059318</td>
          <td>0.041850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.537144</td>
          <td>0.429859</td>
          <td>28.967620</td>
          <td>1.007653</td>
          <td>28.130419</td>
          <td>0.531894</td>
          <td>25.871382</td>
          <td>0.135203</td>
          <td>25.010583</td>
          <td>0.120195</td>
          <td>24.559704</td>
          <td>0.180672</td>
          <td>0.119045</td>
          <td>0.118783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.171402</td>
          <td>1.919121</td>
          <td>31.582417</td>
          <td>2.985892</td>
          <td>27.407293</td>
          <td>0.263361</td>
          <td>26.277494</td>
          <td>0.162632</td>
          <td>25.730084</td>
          <td>0.190357</td>
          <td>26.121052</td>
          <td>0.537071</td>
          <td>0.008205</td>
          <td>0.004428</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.217133</td>
          <td>0.301954</td>
          <td>25.932439</td>
          <td>0.084599</td>
          <td>25.866942</td>
          <td>0.070254</td>
          <td>25.608775</td>
          <td>0.091183</td>
          <td>25.218388</td>
          <td>0.122996</td>
          <td>25.696420</td>
          <td>0.391106</td>
          <td>0.014849</td>
          <td>0.010345</td>
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
          <td>26.497387</td>
          <td>0.377553</td>
          <td>26.493395</td>
          <td>0.138362</td>
          <td>25.444992</td>
          <td>0.048483</td>
          <td>25.108601</td>
          <td>0.058807</td>
          <td>24.774547</td>
          <td>0.083681</td>
          <td>25.049457</td>
          <td>0.233529</td>
          <td>0.021984</td>
          <td>0.018526</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.087189</td>
          <td>0.592484</td>
          <td>26.762309</td>
          <td>0.177063</td>
          <td>25.981614</td>
          <td>0.079575</td>
          <td>25.186371</td>
          <td>0.064337</td>
          <td>24.706171</td>
          <td>0.080358</td>
          <td>24.195863</td>
          <td>0.115141</td>
          <td>0.045849</td>
          <td>0.041437</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.996192</td>
          <td>0.549337</td>
          <td>26.840998</td>
          <td>0.186350</td>
          <td>26.397592</td>
          <td>0.112546</td>
          <td>25.987304</td>
          <td>0.127517</td>
          <td>25.767045</td>
          <td>0.197554</td>
          <td>24.960623</td>
          <td>0.217184</td>
          <td>0.027045</td>
          <td>0.016842</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.572781</td>
          <td>0.431988</td>
          <td>26.131056</td>
          <td>0.112406</td>
          <td>26.031142</td>
          <td>0.092105</td>
          <td>26.131849</td>
          <td>0.163353</td>
          <td>25.958148</td>
          <td>0.259428</td>
          <td>26.076896</td>
          <td>0.579318</td>
          <td>0.107540</td>
          <td>0.099766</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.239048</td>
          <td>0.309591</td>
          <td>26.560859</td>
          <td>0.147673</td>
          <td>26.508902</td>
          <td>0.124857</td>
          <td>26.254033</td>
          <td>0.161588</td>
          <td>25.886917</td>
          <td>0.219873</td>
          <td>25.806394</td>
          <td>0.430039</td>
          <td>0.037790</td>
          <td>0.025562</td>
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
