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

    <pzflow.flow.Flow at 0x7fcef4bead40>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>26.525893</td>
          <td>0.384675</td>
          <td>26.971811</td>
          <td>0.206836</td>
          <td>26.079098</td>
          <td>0.084544</td>
          <td>25.158696</td>
          <td>0.061116</td>
          <td>24.735083</td>
          <td>0.080368</td>
          <td>23.891517</td>
          <td>0.085913</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.993217</td>
          <td>0.210572</td>
          <td>26.694997</td>
          <td>0.144642</td>
          <td>26.288456</td>
          <td>0.164062</td>
          <td>25.741633</td>
          <td>0.192111</td>
          <td>25.411524</td>
          <td>0.311926</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.844911</td>
          <td>0.964542</td>
          <td>29.130016</td>
          <td>1.007597</td>
          <td>27.946498</td>
          <td>0.403894</td>
          <td>25.828438</td>
          <td>0.110274</td>
          <td>25.270107</td>
          <td>0.128356</td>
          <td>24.501590</td>
          <td>0.146188</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.195795</td>
          <td>1.047838</td>
          <td>27.818603</td>
          <td>0.365773</td>
          <td>26.433599</td>
          <td>0.185585</td>
          <td>25.277007</td>
          <td>0.129125</td>
          <td>25.274916</td>
          <td>0.279416</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.406597</td>
          <td>0.350494</td>
          <td>25.973514</td>
          <td>0.087541</td>
          <td>25.941355</td>
          <td>0.074866</td>
          <td>25.842357</td>
          <td>0.111622</td>
          <td>25.423632</td>
          <td>0.146538</td>
          <td>25.150961</td>
          <td>0.252531</td>
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
          <td>26.511847</td>
          <td>0.380511</td>
          <td>26.381071</td>
          <td>0.124964</td>
          <td>25.471242</td>
          <td>0.049346</td>
          <td>25.118301</td>
          <td>0.058965</td>
          <td>24.941489</td>
          <td>0.096374</td>
          <td>24.837761</td>
          <td>0.194616</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.531696</td>
          <td>0.791338</td>
          <td>26.585059</td>
          <td>0.149001</td>
          <td>26.037902</td>
          <td>0.081528</td>
          <td>25.228495</td>
          <td>0.065018</td>
          <td>24.843460</td>
          <td>0.088421</td>
          <td>24.129961</td>
          <td>0.105909</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.147902</td>
          <td>0.609753</td>
          <td>27.073812</td>
          <td>0.225195</td>
          <td>26.327899</td>
          <td>0.105184</td>
          <td>26.282074</td>
          <td>0.163171</td>
          <td>25.684505</td>
          <td>0.183063</td>
          <td>26.699853</td>
          <td>0.800226</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.020039</td>
          <td>0.257022</td>
          <td>26.271261</td>
          <td>0.113596</td>
          <td>26.034484</td>
          <td>0.081283</td>
          <td>25.985914</td>
          <td>0.126464</td>
          <td>25.671908</td>
          <td>0.181121</td>
          <td>25.311862</td>
          <td>0.287903</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.583204</td>
          <td>0.402064</td>
          <td>26.927383</td>
          <td>0.199274</td>
          <td>26.453827</td>
          <td>0.117396</td>
          <td>26.518365</td>
          <td>0.199328</td>
          <td>26.032275</td>
          <td>0.244776</td>
          <td>24.927192</td>
          <td>0.209783</td>
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
          <td>1.398944</td>
          <td>27.475508</td>
          <td>0.831975</td>
          <td>26.610422</td>
          <td>0.174948</td>
          <td>25.947550</td>
          <td>0.088547</td>
          <td>25.287224</td>
          <td>0.081157</td>
          <td>24.560395</td>
          <td>0.081029</td>
          <td>23.893655</td>
          <td>0.101722</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.853324</td>
          <td>0.543571</td>
          <td>27.409224</td>
          <td>0.337417</td>
          <td>26.423820</td>
          <td>0.134188</td>
          <td>26.061113</td>
          <td>0.159175</td>
          <td>25.953009</td>
          <td>0.266685</td>
          <td>25.867328</td>
          <td>0.512428</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.507771</td>
          <td>0.371121</td>
          <td>28.486260</td>
          <td>0.695695</td>
          <td>25.871566</td>
          <td>0.138323</td>
          <td>25.171467</td>
          <td>0.141210</td>
          <td>24.186655</td>
          <td>0.134260</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.106590</td>
          <td>0.677204</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.746626</td>
          <td>0.422949</td>
          <td>26.284530</td>
          <td>0.205567</td>
          <td>25.282894</td>
          <td>0.162314</td>
          <td>25.397079</td>
          <td>0.381018</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.772935</td>
          <td>0.512691</td>
          <td>26.140435</td>
          <td>0.116846</td>
          <td>25.759591</td>
          <td>0.075050</td>
          <td>25.836509</td>
          <td>0.131231</td>
          <td>26.101871</td>
          <td>0.300884</td>
          <td>25.263692</td>
          <td>0.322673</td>
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
          <td>27.432482</td>
          <td>0.818779</td>
          <td>26.569883</td>
          <td>0.172158</td>
          <td>25.425389</td>
          <td>0.057003</td>
          <td>25.096128</td>
          <td>0.070068</td>
          <td>24.941884</td>
          <td>0.115601</td>
          <td>24.563436</td>
          <td>0.185045</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.151673</td>
          <td>0.672468</td>
          <td>26.753133</td>
          <td>0.198063</td>
          <td>26.081992</td>
          <td>0.100054</td>
          <td>25.129723</td>
          <td>0.070925</td>
          <td>24.759667</td>
          <td>0.096951</td>
          <td>24.042607</td>
          <td>0.116338</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.843187</td>
          <td>0.215156</td>
          <td>26.447686</td>
          <td>0.138684</td>
          <td>26.367327</td>
          <td>0.208850</td>
          <td>25.659765</td>
          <td>0.211811</td>
          <td>26.023742</td>
          <td>0.579978</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.231604</td>
          <td>0.346648</td>
          <td>26.270132</td>
          <td>0.134415</td>
          <td>26.143424</td>
          <td>0.108480</td>
          <td>25.735635</td>
          <td>0.124127</td>
          <td>25.599738</td>
          <td>0.205038</td>
          <td>25.124312</td>
          <td>0.297120</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.210866</td>
          <td>0.336041</td>
          <td>27.070936</td>
          <td>0.259090</td>
          <td>26.677869</td>
          <td>0.168473</td>
          <td>26.188554</td>
          <td>0.179171</td>
          <td>25.698459</td>
          <td>0.218190</td>
          <td>26.386973</td>
          <td>0.743608</td>
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
          <td>1.398944</td>
          <td>26.519192</td>
          <td>0.382714</td>
          <td>26.793599</td>
          <td>0.178024</td>
          <td>26.066125</td>
          <td>0.083594</td>
          <td>25.294530</td>
          <td>0.068945</td>
          <td>24.629168</td>
          <td>0.073199</td>
          <td>23.933345</td>
          <td>0.089147</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.911750</td>
          <td>0.196828</td>
          <td>26.521792</td>
          <td>0.124654</td>
          <td>26.629487</td>
          <td>0.218956</td>
          <td>26.027246</td>
          <td>0.243979</td>
          <td>25.179213</td>
          <td>0.258686</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.221402</td>
          <td>0.672116</td>
          <td>32.082813</td>
          <td>3.538294</td>
          <td>28.254856</td>
          <td>0.545101</td>
          <td>25.937692</td>
          <td>0.131962</td>
          <td>25.015629</td>
          <td>0.111559</td>
          <td>24.335081</td>
          <td>0.137631</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.056195</td>
          <td>2.869300</td>
          <td>27.964511</td>
          <td>0.540573</td>
          <td>27.190416</td>
          <td>0.271773</td>
          <td>26.064481</td>
          <td>0.170106</td>
          <td>25.515986</td>
          <td>0.197076</td>
          <td>25.731163</td>
          <td>0.489445</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.732135</td>
          <td>0.450682</td>
          <td>26.162827</td>
          <td>0.103471</td>
          <td>26.015030</td>
          <td>0.080014</td>
          <td>25.704120</td>
          <td>0.099060</td>
          <td>25.419641</td>
          <td>0.146239</td>
          <td>25.284542</td>
          <td>0.281990</td>
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
          <td>28.610465</td>
          <td>1.524113</td>
          <td>26.320185</td>
          <td>0.126895</td>
          <td>25.398092</td>
          <td>0.050083</td>
          <td>25.068283</td>
          <td>0.061307</td>
          <td>24.944644</td>
          <td>0.104528</td>
          <td>24.554100</td>
          <td>0.165561</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.646343</td>
          <td>0.159236</td>
          <td>25.956007</td>
          <td>0.077110</td>
          <td>25.108302</td>
          <td>0.059475</td>
          <td>24.750924</td>
          <td>0.082857</td>
          <td>24.062425</td>
          <td>0.101550</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.832951</td>
          <td>0.499821</td>
          <td>26.804291</td>
          <td>0.187165</td>
          <td>26.313844</td>
          <td>0.109082</td>
          <td>26.080557</td>
          <td>0.144327</td>
          <td>26.482775</td>
          <td>0.367552</td>
          <td>25.545928</td>
          <td>0.363137</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.620692</td>
          <td>0.444260</td>
          <td>26.342966</td>
          <td>0.133553</td>
          <td>26.228727</td>
          <td>0.108112</td>
          <td>25.943289</td>
          <td>0.137144</td>
          <td>25.638683</td>
          <td>0.196593</td>
          <td>25.379188</td>
          <td>0.338462</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.581540</td>
          <td>0.411224</td>
          <td>26.634339</td>
          <td>0.160643</td>
          <td>26.502668</td>
          <td>0.127287</td>
          <td>26.048284</td>
          <td>0.138939</td>
          <td>25.784196</td>
          <td>0.206643</td>
          <td>25.242053</td>
          <td>0.282395</td>
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
