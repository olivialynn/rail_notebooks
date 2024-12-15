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

    <pzflow.flow.Flow at 0x7f5b10cccd00>



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
          <td>28.691892</td>
          <td>1.537667</td>
          <td>26.434189</td>
          <td>0.130844</td>
          <td>26.081539</td>
          <td>0.084726</td>
          <td>25.189040</td>
          <td>0.062783</td>
          <td>24.886135</td>
          <td>0.091802</td>
          <td>24.650806</td>
          <td>0.166107</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.198836</td>
          <td>1.941530</td>
          <td>28.815419</td>
          <td>0.828492</td>
          <td>27.630560</td>
          <td>0.315263</td>
          <td>27.804326</td>
          <td>0.549058</td>
          <td>26.051849</td>
          <td>0.248750</td>
          <td>27.145053</td>
          <td>1.054828</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.158166</td>
          <td>0.287554</td>
          <td>25.895891</td>
          <td>0.081762</td>
          <td>24.788196</td>
          <td>0.026998</td>
          <td>23.882648</td>
          <td>0.019936</td>
          <td>23.161434</td>
          <td>0.020145</td>
          <td>22.845848</td>
          <td>0.034031</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.504046</td>
          <td>2.019058</td>
          <td>27.575960</td>
          <td>0.301770</td>
          <td>26.668681</td>
          <td>0.225999</td>
          <td>25.660191</td>
          <td>0.179332</td>
          <td>25.539433</td>
          <td>0.345280</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.187872</td>
          <td>0.294522</td>
          <td>25.693241</td>
          <td>0.068374</td>
          <td>25.439126</td>
          <td>0.047959</td>
          <td>24.806707</td>
          <td>0.044714</td>
          <td>24.380741</td>
          <td>0.058730</td>
          <td>23.767460</td>
          <td>0.077009</td>
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
          <td>26.059432</td>
          <td>0.265422</td>
          <td>26.331529</td>
          <td>0.119707</td>
          <td>26.080510</td>
          <td>0.084649</td>
          <td>26.120352</td>
          <td>0.142043</td>
          <td>25.600574</td>
          <td>0.170480</td>
          <td>25.455632</td>
          <td>0.323100</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.788295</td>
          <td>0.469670</td>
          <td>26.822072</td>
          <td>0.182348</td>
          <td>26.832595</td>
          <td>0.162745</td>
          <td>26.294841</td>
          <td>0.164958</td>
          <td>25.819838</td>
          <td>0.205162</td>
          <td>25.190118</td>
          <td>0.260765</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.366895</td>
          <td>0.286356</td>
          <td>26.773909</td>
          <td>0.154780</td>
          <td>26.272446</td>
          <td>0.161835</td>
          <td>27.182008</td>
          <td>0.594276</td>
          <td>25.409192</td>
          <td>0.311345</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.554824</td>
          <td>0.803361</td>
          <td>27.630505</td>
          <td>0.353333</td>
          <td>26.660993</td>
          <td>0.140468</td>
          <td>25.917161</td>
          <td>0.119135</td>
          <td>25.299704</td>
          <td>0.131686</td>
          <td>26.162431</td>
          <td>0.553122</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.997364</td>
          <td>0.547662</td>
          <td>26.659341</td>
          <td>0.158786</td>
          <td>26.025047</td>
          <td>0.080609</td>
          <td>25.676927</td>
          <td>0.096582</td>
          <td>25.323339</td>
          <td>0.134405</td>
          <td>25.427502</td>
          <td>0.315935</td>
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
          <td>28.564926</td>
          <td>1.542529</td>
          <td>26.691192</td>
          <td>0.187323</td>
          <td>25.921731</td>
          <td>0.086558</td>
          <td>25.300415</td>
          <td>0.082107</td>
          <td>25.068149</td>
          <td>0.126350</td>
          <td>24.740891</td>
          <td>0.210428</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.049293</td>
          <td>0.548228</td>
          <td>27.912739</td>
          <td>0.452788</td>
          <td>27.204361</td>
          <td>0.404900</td>
          <td>26.879910</td>
          <td>0.545972</td>
          <td>25.692127</td>
          <td>0.449813</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.891211</td>
          <td>0.566399</td>
          <td>25.917288</td>
          <td>0.098100</td>
          <td>24.756555</td>
          <td>0.031580</td>
          <td>23.888054</td>
          <td>0.024177</td>
          <td>23.156716</td>
          <td>0.024036</td>
          <td>22.952565</td>
          <td>0.045323</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.067965</td>
          <td>1.227644</td>
          <td>27.939670</td>
          <td>0.532339</td>
          <td>27.354560</td>
          <td>0.311290</td>
          <td>26.475996</td>
          <td>0.241041</td>
          <td>26.374115</td>
          <td>0.395871</td>
          <td>25.097523</td>
          <td>0.300697</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.914542</td>
          <td>0.263069</td>
          <td>25.813688</td>
          <td>0.087818</td>
          <td>25.543585</td>
          <td>0.061990</td>
          <td>24.812015</td>
          <td>0.053305</td>
          <td>24.356853</td>
          <td>0.067714</td>
          <td>23.592389</td>
          <td>0.078080</td>
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
          <td>27.520800</td>
          <td>0.866411</td>
          <td>26.029816</td>
          <td>0.108105</td>
          <td>26.480119</td>
          <td>0.143766</td>
          <td>26.511287</td>
          <td>0.237307</td>
          <td>27.044874</td>
          <td>0.624429</td>
          <td>25.611645</td>
          <td>0.431088</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.945298</td>
          <td>1.110422</td>
          <td>27.160453</td>
          <td>0.277330</td>
          <td>26.698642</td>
          <td>0.170506</td>
          <td>26.621219</td>
          <td>0.255580</td>
          <td>25.709162</td>
          <td>0.218925</td>
          <td>25.547864</td>
          <td>0.404462</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.350998</td>
          <td>1.392292</td>
          <td>27.879803</td>
          <td>0.488862</td>
          <td>26.539194</td>
          <td>0.150042</td>
          <td>26.598693</td>
          <td>0.252997</td>
          <td>26.014218</td>
          <td>0.283565</td>
          <td>25.674816</td>
          <td>0.448928</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.755586</td>
          <td>1.711690</td>
          <td>27.193595</td>
          <td>0.291405</td>
          <td>26.389381</td>
          <td>0.134316</td>
          <td>26.056827</td>
          <td>0.163649</td>
          <td>25.746915</td>
          <td>0.231786</td>
          <td>25.585547</td>
          <td>0.426544</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.219556</td>
          <td>0.338356</td>
          <td>26.670185</td>
          <td>0.185639</td>
          <td>26.313853</td>
          <td>0.123189</td>
          <td>25.644787</td>
          <td>0.112214</td>
          <td>25.094473</td>
          <td>0.130548</td>
          <td>24.776355</td>
          <td>0.218882</td>
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
          <td>26.323670</td>
          <td>0.328308</td>
          <td>27.029755</td>
          <td>0.217115</td>
          <td>26.040297</td>
          <td>0.081712</td>
          <td>25.212415</td>
          <td>0.064107</td>
          <td>24.857650</td>
          <td>0.089543</td>
          <td>24.996085</td>
          <td>0.222220</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.742019</td>
          <td>1.576405</td>
          <td>28.307079</td>
          <td>0.587282</td>
          <td>26.997879</td>
          <td>0.187440</td>
          <td>27.866554</td>
          <td>0.574642</td>
          <td>27.336179</td>
          <td>0.662420</td>
          <td>26.211067</td>
          <td>0.573250</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.679289</td>
          <td>0.204882</td>
          <td>26.008931</td>
          <td>0.097046</td>
          <td>24.755833</td>
          <td>0.028486</td>
          <td>23.880717</td>
          <td>0.021639</td>
          <td>23.137696</td>
          <td>0.021385</td>
          <td>22.873001</td>
          <td>0.037974</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.521573</td>
          <td>0.887615</td>
          <td>29.063549</td>
          <td>1.108104</td>
          <td>27.692708</td>
          <td>0.404590</td>
          <td>26.652632</td>
          <td>0.277581</td>
          <td>26.028303</td>
          <td>0.300480</td>
          <td>25.745955</td>
          <td>0.494833</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.098715</td>
          <td>0.589376</td>
          <td>25.767893</td>
          <td>0.073126</td>
          <td>25.371159</td>
          <td>0.045216</td>
          <td>24.863005</td>
          <td>0.047077</td>
          <td>24.282291</td>
          <td>0.053894</td>
          <td>23.755998</td>
          <td>0.076347</td>
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
          <td>26.179440</td>
          <td>0.307857</td>
          <td>26.303912</td>
          <td>0.125119</td>
          <td>26.003916</td>
          <td>0.085632</td>
          <td>25.991572</td>
          <td>0.137818</td>
          <td>25.907346</td>
          <td>0.237717</td>
          <td>25.251464</td>
          <td>0.295529</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.554968</td>
          <td>0.810278</td>
          <td>27.057490</td>
          <td>0.225191</td>
          <td>26.519352</td>
          <td>0.126313</td>
          <td>26.121122</td>
          <td>0.144565</td>
          <td>26.437186</td>
          <td>0.344524</td>
          <td>26.395020</td>
          <td>0.660808</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.241049</td>
          <td>1.241190</td>
          <td>27.100298</td>
          <td>0.239639</td>
          <td>26.780818</td>
          <td>0.163287</td>
          <td>26.650430</td>
          <td>0.233628</td>
          <td>25.651598</td>
          <td>0.186610</td>
          <td>26.160911</td>
          <td>0.575888</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.409292</td>
          <td>1.397537</td>
          <td>26.971853</td>
          <td>0.227611</td>
          <td>26.415240</td>
          <td>0.127156</td>
          <td>25.648444</td>
          <td>0.106156</td>
          <td>25.959156</td>
          <td>0.256551</td>
          <td>25.630526</td>
          <td>0.411626</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.281380</td>
          <td>0.325365</td>
          <td>26.452323</td>
          <td>0.137416</td>
          <td>26.012878</td>
          <td>0.082929</td>
          <td>25.643455</td>
          <td>0.097689</td>
          <td>25.308265</td>
          <td>0.137833</td>
          <td>25.034571</td>
          <td>0.238300</td>
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
