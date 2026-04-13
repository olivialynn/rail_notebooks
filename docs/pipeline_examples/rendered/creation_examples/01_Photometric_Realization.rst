Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fa9e0d6f010>



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
    0      23.994413  0.026085  0.017308  
    1      25.391064  0.068103  0.058547  
    2      24.304707  0.164963  0.082843  
    3      25.291103  0.225824  0.169231  
    4      25.096743  0.125392  0.101629  
    ...          ...       ...       ...  
    99995  24.737946  0.005189  0.003417  
    99996  24.224169  0.191556  0.105173  
    99997  25.613836  0.150812  0.117374  
    99998  25.274899  0.146404  0.102930  
    99999  25.699642  0.075476  0.073162  
    
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

    Inserting handle into data store.  output_truth: None, error_model
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
          <td>26.823585</td>
          <td>0.482178</td>
          <td>26.582728</td>
          <td>0.148703</td>
          <td>26.019293</td>
          <td>0.080201</td>
          <td>25.135956</td>
          <td>0.059896</td>
          <td>24.635822</td>
          <td>0.073621</td>
          <td>24.180456</td>
          <td>0.110683</td>
          <td>0.026085</td>
          <td>0.017308</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.824764</td>
          <td>0.410819</td>
          <td>26.636626</td>
          <td>0.137548</td>
          <td>26.286245</td>
          <td>0.163753</td>
          <td>26.088626</td>
          <td>0.256375</td>
          <td>25.318521</td>
          <td>0.289456</td>
          <td>0.068103</td>
          <td>0.058547</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.505350</td>
          <td>0.777791</td>
          <td>28.896742</td>
          <td>0.872643</td>
          <td>27.494530</td>
          <td>0.282580</td>
          <td>26.140903</td>
          <td>0.144578</td>
          <td>24.983724</td>
          <td>0.100009</td>
          <td>24.502674</td>
          <td>0.146324</td>
          <td>0.164963</td>
          <td>0.082843</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.331707</td>
          <td>0.692503</td>
          <td>31.387424</td>
          <td>2.803934</td>
          <td>27.257108</td>
          <td>0.232624</td>
          <td>26.295095</td>
          <td>0.164994</td>
          <td>25.562159</td>
          <td>0.164992</td>
          <td>25.507162</td>
          <td>0.336589</td>
          <td>0.225824</td>
          <td>0.169231</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.252488</td>
          <td>0.310190</td>
          <td>26.022257</td>
          <td>0.091371</td>
          <td>25.959924</td>
          <td>0.076105</td>
          <td>25.805202</td>
          <td>0.108060</td>
          <td>25.643614</td>
          <td>0.176829</td>
          <td>25.275948</td>
          <td>0.279650</td>
          <td>0.125392</td>
          <td>0.101629</td>
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
          <td>26.401588</td>
          <td>0.349116</td>
          <td>26.508980</td>
          <td>0.139567</td>
          <td>25.469878</td>
          <td>0.049286</td>
          <td>25.078569</td>
          <td>0.056922</td>
          <td>24.887065</td>
          <td>0.091877</td>
          <td>24.615817</td>
          <td>0.161221</td>
          <td>0.005189</td>
          <td>0.003417</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.181869</td>
          <td>0.624467</td>
          <td>26.596027</td>
          <td>0.150409</td>
          <td>26.150319</td>
          <td>0.090014</td>
          <td>25.262288</td>
          <td>0.066995</td>
          <td>24.825550</td>
          <td>0.087038</td>
          <td>24.605817</td>
          <td>0.159850</td>
          <td>0.191556</td>
          <td>0.105173</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.124486</td>
          <td>0.599762</td>
          <td>26.630889</td>
          <td>0.154969</td>
          <td>26.462804</td>
          <td>0.118317</td>
          <td>26.159330</td>
          <td>0.146887</td>
          <td>25.906612</td>
          <td>0.220584</td>
          <td>25.231710</td>
          <td>0.269771</td>
          <td>0.150812</td>
          <td>0.117374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.442279</td>
          <td>0.746010</td>
          <td>26.289255</td>
          <td>0.115389</td>
          <td>25.976942</td>
          <td>0.077257</td>
          <td>25.957576</td>
          <td>0.123393</td>
          <td>25.536049</td>
          <td>0.161356</td>
          <td>25.509898</td>
          <td>0.337319</td>
          <td>0.146404</td>
          <td>0.102930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.733975</td>
          <td>0.450927</td>
          <td>26.943222</td>
          <td>0.201940</td>
          <td>26.507150</td>
          <td>0.122965</td>
          <td>26.392571</td>
          <td>0.179251</td>
          <td>25.929672</td>
          <td>0.224855</td>
          <td>26.583269</td>
          <td>0.741007</td>
          <td>0.075476</td>
          <td>0.073162</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


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
          <td>27.083998</td>
          <td>0.260031</td>
          <td>26.045271</td>
          <td>0.096641</td>
          <td>25.283591</td>
          <td>0.081035</td>
          <td>24.770328</td>
          <td>0.097614</td>
          <td>23.811326</td>
          <td>0.094801</td>
          <td>0.026085</td>
          <td>0.017308</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.433651</td>
          <td>0.815953</td>
          <td>27.360152</td>
          <td>0.328083</td>
          <td>26.511870</td>
          <td>0.146667</td>
          <td>26.073264</td>
          <td>0.163005</td>
          <td>25.662181</td>
          <td>0.212386</td>
          <td>24.855315</td>
          <td>0.234499</td>
          <td>0.068103</td>
          <td>0.058547</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.618115</td>
          <td>0.839578</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.014347</td>
          <td>0.161397</td>
          <td>24.913782</td>
          <td>0.116532</td>
          <td>24.201020</td>
          <td>0.140312</td>
          <td>0.164963</td>
          <td>0.082843</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.133901</td>
          <td>0.635169</td>
          <td>26.963640</td>
          <td>0.237523</td>
          <td>26.046146</td>
          <td>0.176887</td>
          <td>25.437497</td>
          <td>0.194444</td>
          <td>26.016273</td>
          <td>0.629331</td>
          <td>0.225824</td>
          <td>0.169231</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.492718</td>
          <td>0.427437</td>
          <td>26.098500</td>
          <td>0.116994</td>
          <td>25.983179</td>
          <td>0.095320</td>
          <td>25.706964</td>
          <td>0.122436</td>
          <td>25.343706</td>
          <td>0.166891</td>
          <td>25.133992</td>
          <td>0.302531</td>
          <td>0.125392</td>
          <td>0.101629</td>
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
          <td>26.415047</td>
          <td>0.391501</td>
          <td>26.576887</td>
          <td>0.170041</td>
          <td>25.418019</td>
          <td>0.055442</td>
          <td>25.051599</td>
          <td>0.065901</td>
          <td>24.973112</td>
          <td>0.116345</td>
          <td>25.241296</td>
          <td>0.316878</td>
          <td>0.005189</td>
          <td>0.003417</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.205160</td>
          <td>0.726994</td>
          <td>27.029263</td>
          <td>0.264279</td>
          <td>25.853913</td>
          <td>0.087798</td>
          <td>25.262910</td>
          <td>0.085742</td>
          <td>24.764773</td>
          <td>0.104373</td>
          <td>24.032797</td>
          <td>0.123778</td>
          <td>0.191556</td>
          <td>0.105173</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.999295</td>
          <td>0.293633</td>
          <td>26.395648</td>
          <td>0.153356</td>
          <td>26.210676</td>
          <td>0.118136</td>
          <td>26.031936</td>
          <td>0.164585</td>
          <td>26.071999</td>
          <td>0.309706</td>
          <td>25.758307</td>
          <td>0.497106</td>
          <td>0.150812</td>
          <td>0.117374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.646718</td>
          <td>0.952930</td>
          <td>26.416999</td>
          <td>0.155228</td>
          <td>26.158390</td>
          <td>0.112108</td>
          <td>25.832550</td>
          <td>0.137748</td>
          <td>25.917212</td>
          <td>0.271592</td>
          <td>25.400383</td>
          <td>0.376580</td>
          <td>0.146404</td>
          <td>0.102930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.205501</td>
          <td>0.291060</td>
          <td>26.650879</td>
          <td>0.165991</td>
          <td>26.295999</td>
          <td>0.197825</td>
          <td>26.399628</td>
          <td>0.386847</td>
          <td>26.434855</td>
          <td>0.772593</td>
          <td>0.075476</td>
          <td>0.073162</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


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
          <td>27.308842</td>
          <td>0.684221</td>
          <td>26.905733</td>
          <td>0.196758</td>
          <td>25.966825</td>
          <td>0.077081</td>
          <td>25.212684</td>
          <td>0.064565</td>
          <td>24.806215</td>
          <td>0.086137</td>
          <td>23.922427</td>
          <td>0.088893</td>
          <td>0.026085</td>
          <td>0.017308</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.751645</td>
          <td>0.934368</td>
          <td>27.637481</td>
          <td>0.369874</td>
          <td>26.618925</td>
          <td>0.142535</td>
          <td>26.180514</td>
          <td>0.157723</td>
          <td>26.024238</td>
          <td>0.255228</td>
          <td>25.580569</td>
          <td>0.374099</td>
          <td>0.068103</td>
          <td>0.058547</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.764329</td>
          <td>1.569265</td>
          <td>28.194933</td>
          <td>0.565601</td>
          <td>26.133823</td>
          <td>0.172356</td>
          <td>24.880566</td>
          <td>0.109265</td>
          <td>24.162419</td>
          <td>0.130878</td>
          <td>0.164963</td>
          <td>0.082843</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.598836</td>
          <td>0.457034</td>
          <td>29.967122</td>
          <td>1.791541</td>
          <td>26.158195</td>
          <td>0.209545</td>
          <td>25.714773</td>
          <td>0.263058</td>
          <td>25.296110</td>
          <td>0.395250</td>
          <td>0.225824</td>
          <td>0.169231</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.605710</td>
          <td>0.202085</td>
          <td>26.243152</td>
          <td>0.126387</td>
          <td>25.938105</td>
          <td>0.086793</td>
          <td>25.642587</td>
          <td>0.109522</td>
          <td>25.272325</td>
          <td>0.148968</td>
          <td>24.975486</td>
          <td>0.252694</td>
          <td>0.125392</td>
          <td>0.101629</td>
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
          <td>27.933461</td>
          <td>1.017642</td>
          <td>26.522146</td>
          <td>0.141190</td>
          <td>25.431439</td>
          <td>0.047645</td>
          <td>25.152248</td>
          <td>0.060785</td>
          <td>24.821301</td>
          <td>0.086736</td>
          <td>24.959994</td>
          <td>0.215666</td>
          <td>0.005189</td>
          <td>0.003417</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.883013</td>
          <td>1.111633</td>
          <td>26.700655</td>
          <td>0.201653</td>
          <td>25.972191</td>
          <td>0.097497</td>
          <td>25.281000</td>
          <td>0.087174</td>
          <td>24.790121</td>
          <td>0.106809</td>
          <td>24.090137</td>
          <td>0.130171</td>
          <td>0.191556</td>
          <td>0.105173</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.275285</td>
          <td>0.745384</td>
          <td>26.740706</td>
          <td>0.201623</td>
          <td>26.230377</td>
          <td>0.117631</td>
          <td>26.096063</td>
          <td>0.170096</td>
          <td>25.773314</td>
          <td>0.238077</td>
          <td>25.727621</td>
          <td>0.476899</td>
          <td>0.150812</td>
          <td>0.117374</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.715994</td>
          <td>0.496528</td>
          <td>26.279685</td>
          <td>0.133608</td>
          <td>25.886695</td>
          <td>0.085214</td>
          <td>25.897773</td>
          <td>0.140446</td>
          <td>25.680002</td>
          <td>0.215901</td>
          <td>25.023095</td>
          <td>0.269535</td>
          <td>0.146404</td>
          <td>0.102930</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.077993</td>
          <td>0.282033</td>
          <td>26.790206</td>
          <td>0.188235</td>
          <td>26.535551</td>
          <td>0.134999</td>
          <td>26.356246</td>
          <td>0.186487</td>
          <td>25.727991</td>
          <td>0.203013</td>
          <td>27.120988</td>
          <td>1.091330</td>
          <td>0.075476</td>
          <td>0.073162</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


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
