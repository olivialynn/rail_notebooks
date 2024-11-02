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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f0994373fa0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>26.534915</td>
          <td>0.387370</td>
          <td>26.428101</td>
          <td>0.130157</td>
          <td>26.045919</td>
          <td>0.082107</td>
          <td>25.275072</td>
          <td>0.067758</td>
          <td>24.895985</td>
          <td>0.092600</td>
          <td>24.522875</td>
          <td>0.148886</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.605696</td>
          <td>0.346504</td>
          <td>27.664397</td>
          <td>0.323884</td>
          <td>27.050610</td>
          <td>0.308686</td>
          <td>27.143867</td>
          <td>0.578367</td>
          <td>26.435362</td>
          <td>0.670377</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.237670</td>
          <td>0.649205</td>
          <td>25.971789</td>
          <td>0.087409</td>
          <td>24.751849</td>
          <td>0.026156</td>
          <td>23.845385</td>
          <td>0.019317</td>
          <td>23.116277</td>
          <td>0.019390</td>
          <td>22.813609</td>
          <td>0.033078</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.795466</td>
          <td>0.935722</td>
          <td>27.384813</td>
          <td>0.290531</td>
          <td>27.828830</td>
          <td>0.368706</td>
          <td>26.646346</td>
          <td>0.221842</td>
          <td>26.135543</td>
          <td>0.266402</td>
          <td>25.492805</td>
          <td>0.332784</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.600084</td>
          <td>0.407308</td>
          <td>25.642138</td>
          <td>0.065353</td>
          <td>25.416379</td>
          <td>0.047000</td>
          <td>24.793235</td>
          <td>0.044183</td>
          <td>24.327900</td>
          <td>0.056040</td>
          <td>23.710466</td>
          <td>0.073226</td>
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
          <td>2.147172</td>
          <td>26.754698</td>
          <td>0.458005</td>
          <td>26.638687</td>
          <td>0.156006</td>
          <td>26.112088</td>
          <td>0.087036</td>
          <td>26.074529</td>
          <td>0.136540</td>
          <td>25.682510</td>
          <td>0.182754</td>
          <td>25.795133</td>
          <td>0.421061</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.129637</td>
          <td>0.601949</td>
          <td>26.974812</td>
          <td>0.207356</td>
          <td>27.044225</td>
          <td>0.194734</td>
          <td>26.625375</td>
          <td>0.218002</td>
          <td>26.424231</td>
          <td>0.336009</td>
          <td>25.609452</td>
          <td>0.364792</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.950951</td>
          <td>1.028183</td>
          <td>27.419706</td>
          <td>0.298815</td>
          <td>26.794152</td>
          <td>0.157485</td>
          <td>26.527074</td>
          <td>0.200791</td>
          <td>25.834767</td>
          <td>0.207743</td>
          <td>25.364305</td>
          <td>0.300335</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.188124</td>
          <td>1.932618</td>
          <td>27.713603</td>
          <td>0.377039</td>
          <td>26.801685</td>
          <td>0.158503</td>
          <td>25.778262</td>
          <td>0.105546</td>
          <td>25.644389</td>
          <td>0.176945</td>
          <td>25.895768</td>
          <td>0.454422</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.587404</td>
          <td>0.403363</td>
          <td>26.533743</td>
          <td>0.142574</td>
          <td>26.097387</td>
          <td>0.085917</td>
          <td>25.622675</td>
          <td>0.092089</td>
          <td>25.172583</td>
          <td>0.117937</td>
          <td>25.215963</td>
          <td>0.266330</td>
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
          <td>0.890625</td>
          <td>27.635620</td>
          <td>0.920545</td>
          <td>26.520008</td>
          <td>0.161997</td>
          <td>25.941210</td>
          <td>0.088055</td>
          <td>25.308442</td>
          <td>0.082690</td>
          <td>25.109061</td>
          <td>0.130906</td>
          <td>25.414113</td>
          <td>0.363239</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.455216</td>
          <td>0.727590</td>
          <td>27.476432</td>
          <td>0.322861</td>
          <td>27.499158</td>
          <td>0.505466</td>
          <td>27.633384</td>
          <td>0.907952</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.255561</td>
          <td>0.351039</td>
          <td>25.867246</td>
          <td>0.093893</td>
          <td>24.781571</td>
          <td>0.032282</td>
          <td>23.840944</td>
          <td>0.023215</td>
          <td>23.159792</td>
          <td>0.024100</td>
          <td>22.860703</td>
          <td>0.041780</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.423319</td>
          <td>0.412255</td>
          <td>27.833828</td>
          <td>0.492560</td>
          <td>26.991351</td>
          <td>0.231556</td>
          <td>26.847173</td>
          <td>0.325648</td>
          <td>26.219384</td>
          <td>0.350922</td>
          <td>25.198877</td>
          <td>0.326072</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>30.485418</td>
          <td>3.220336</td>
          <td>25.582309</td>
          <td>0.071629</td>
          <td>25.411097</td>
          <td>0.055118</td>
          <td>24.775855</td>
          <td>0.051621</td>
          <td>24.432351</td>
          <td>0.072391</td>
          <td>23.641979</td>
          <td>0.081571</td>
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
          <td>2.147172</td>
          <td>26.732384</td>
          <td>0.504252</td>
          <td>26.514391</td>
          <td>0.164220</td>
          <td>26.070762</td>
          <td>0.100753</td>
          <td>26.161192</td>
          <td>0.176988</td>
          <td>26.310738</td>
          <td>0.361798</td>
          <td>26.024345</td>
          <td>0.584206</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.104486</td>
          <td>0.650955</td>
          <td>26.894424</td>
          <td>0.222882</td>
          <td>26.744721</td>
          <td>0.177312</td>
          <td>26.419796</td>
          <td>0.216365</td>
          <td>26.197907</td>
          <td>0.326030</td>
          <td>25.021184</td>
          <td>0.266336</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.103492</td>
          <td>0.309133</td>
          <td>28.081299</td>
          <td>0.566249</td>
          <td>26.992112</td>
          <td>0.220102</td>
          <td>26.558814</td>
          <td>0.244837</td>
          <td>26.338072</td>
          <td>0.366926</td>
          <td>29.327292</td>
          <td>2.996275</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.458867</td>
          <td>0.837707</td>
          <td>27.488433</td>
          <td>0.368250</td>
          <td>26.447904</td>
          <td>0.141269</td>
          <td>25.642031</td>
          <td>0.114428</td>
          <td>25.464261</td>
          <td>0.182933</td>
          <td>25.702619</td>
          <td>0.465956</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.340211</td>
          <td>1.382789</td>
          <td>26.581796</td>
          <td>0.172250</td>
          <td>26.094816</td>
          <td>0.101778</td>
          <td>25.697648</td>
          <td>0.117500</td>
          <td>25.387295</td>
          <td>0.167864</td>
          <td>24.899050</td>
          <td>0.242308</td>
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
          <td>0.890625</td>
          <td>26.710846</td>
          <td>0.443166</td>
          <td>26.680803</td>
          <td>0.161740</td>
          <td>26.167418</td>
          <td>0.091390</td>
          <td>25.334739</td>
          <td>0.071443</td>
          <td>25.146818</td>
          <td>0.115337</td>
          <td>25.196735</td>
          <td>0.262214</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.993781</td>
          <td>0.927833</td>
          <td>27.899537</td>
          <td>0.389854</td>
          <td>27.580521</td>
          <td>0.466085</td>
          <td>26.691131</td>
          <td>0.413986</td>
          <td>26.542532</td>
          <td>0.721595</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.597138</td>
          <td>0.427598</td>
          <td>25.955794</td>
          <td>0.092630</td>
          <td>24.811310</td>
          <td>0.029904</td>
          <td>23.836815</td>
          <td>0.020843</td>
          <td>23.134572</td>
          <td>0.021328</td>
          <td>22.876850</td>
          <td>0.038103</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.521542</td>
          <td>0.793862</td>
          <td>27.169925</td>
          <td>0.267274</td>
          <td>26.311893</td>
          <td>0.209593</td>
          <td>25.722690</td>
          <td>0.234165</td>
          <td>25.286858</td>
          <td>0.348426</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.207137</td>
          <td>0.299389</td>
          <td>25.783491</td>
          <td>0.074140</td>
          <td>25.349686</td>
          <td>0.044362</td>
          <td>24.746390</td>
          <td>0.042448</td>
          <td>24.406418</td>
          <td>0.060170</td>
          <td>23.750111</td>
          <td>0.075951</td>
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
          <td>2.147172</td>
          <td>27.835196</td>
          <td>0.996515</td>
          <td>26.754653</td>
          <td>0.184067</td>
          <td>25.984395</td>
          <td>0.084172</td>
          <td>26.050865</td>
          <td>0.145040</td>
          <td>25.933737</td>
          <td>0.242951</td>
          <td>26.519622</td>
          <td>0.755497</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.020548</td>
          <td>0.218378</td>
          <td>26.835110</td>
          <td>0.165725</td>
          <td>26.309692</td>
          <td>0.169882</td>
          <td>26.186055</td>
          <td>0.281830</td>
          <td>25.101978</td>
          <td>0.246478</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.134509</td>
          <td>1.920611</td>
          <td>27.468221</td>
          <td>0.322957</td>
          <td>26.945074</td>
          <td>0.187722</td>
          <td>26.558783</td>
          <td>0.216499</td>
          <td>26.036171</td>
          <td>0.257033</td>
          <td>25.526285</td>
          <td>0.357593</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.399343</td>
          <td>1.390346</td>
          <td>27.193835</td>
          <td>0.273139</td>
          <td>26.393335</td>
          <td>0.124765</td>
          <td>25.706588</td>
          <td>0.111683</td>
          <td>25.868412</td>
          <td>0.238095</td>
          <td>25.748757</td>
          <td>0.450319</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>25.792933</td>
          <td>0.218650</td>
          <td>26.637469</td>
          <td>0.161073</td>
          <td>26.270416</td>
          <td>0.103980</td>
          <td>25.661527</td>
          <td>0.099249</td>
          <td>25.282387</td>
          <td>0.134788</td>
          <td>25.165858</td>
          <td>0.265427</td>
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
