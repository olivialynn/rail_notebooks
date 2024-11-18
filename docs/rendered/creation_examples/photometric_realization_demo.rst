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

    <pzflow.flow.Flow at 0x7f39dff1b4c0>



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
          <td>27.942767</td>
          <td>1.023184</td>
          <td>26.730550</td>
          <td>0.168725</td>
          <td>26.147622</td>
          <td>0.089801</td>
          <td>25.382041</td>
          <td>0.074485</td>
          <td>24.985808</td>
          <td>0.100192</td>
          <td>24.949368</td>
          <td>0.213707</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.026125</td>
          <td>2.474571</td>
          <td>27.872197</td>
          <td>0.381364</td>
          <td>27.048752</td>
          <td>0.308227</td>
          <td>26.743976</td>
          <td>0.430658</td>
          <td>28.140175</td>
          <td>1.770136</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.814034</td>
          <td>1.631394</td>
          <td>25.820362</td>
          <td>0.076496</td>
          <td>24.766286</td>
          <td>0.026487</td>
          <td>23.857697</td>
          <td>0.019519</td>
          <td>23.120116</td>
          <td>0.019453</td>
          <td>22.855385</td>
          <td>0.034319</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>31.088432</td>
          <td>3.669892</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.346205</td>
          <td>0.250366</td>
          <td>26.902129</td>
          <td>0.273816</td>
          <td>25.611477</td>
          <td>0.172068</td>
          <td>25.820960</td>
          <td>0.429426</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.363761</td>
          <td>0.338865</td>
          <td>25.684669</td>
          <td>0.067858</td>
          <td>25.533385</td>
          <td>0.052145</td>
          <td>24.803590</td>
          <td>0.044591</td>
          <td>24.347469</td>
          <td>0.057022</td>
          <td>23.619870</td>
          <td>0.067585</td>
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
          <td>26.356609</td>
          <td>0.336956</td>
          <td>26.316337</td>
          <td>0.118138</td>
          <td>26.312755</td>
          <td>0.103800</td>
          <td>26.212400</td>
          <td>0.153732</td>
          <td>25.917293</td>
          <td>0.222553</td>
          <td>25.605807</td>
          <td>0.363754</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.071892</td>
          <td>1.103766</td>
          <td>27.272978</td>
          <td>0.265322</td>
          <td>26.585074</td>
          <td>0.131556</td>
          <td>26.711040</td>
          <td>0.234077</td>
          <td>25.869056</td>
          <td>0.213785</td>
          <td>25.273048</td>
          <td>0.278993</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.305730</td>
          <td>0.680340</td>
          <td>27.332821</td>
          <td>0.278561</td>
          <td>27.321469</td>
          <td>0.245323</td>
          <td>26.410752</td>
          <td>0.182033</td>
          <td>26.702908</td>
          <td>0.417388</td>
          <td>25.416879</td>
          <td>0.313265</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.417038</td>
          <td>0.353379</td>
          <td>27.705676</td>
          <td>0.374721</td>
          <td>26.614036</td>
          <td>0.134891</td>
          <td>25.623221</td>
          <td>0.092133</td>
          <td>25.694787</td>
          <td>0.184662</td>
          <td>25.821542</td>
          <td>0.429616</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.890632</td>
          <td>0.506671</td>
          <td>26.478980</td>
          <td>0.136004</td>
          <td>26.083003</td>
          <td>0.084835</td>
          <td>25.823421</td>
          <td>0.109793</td>
          <td>25.112448</td>
          <td>0.111919</td>
          <td>24.776872</td>
          <td>0.184871</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.528783</td>
          <td>0.163213</td>
          <td>26.150837</td>
          <td>0.105825</td>
          <td>25.247509</td>
          <td>0.078363</td>
          <td>24.939289</td>
          <td>0.112965</td>
          <td>25.540425</td>
          <td>0.400649</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.741247</td>
          <td>0.436395</td>
          <td>27.327104</td>
          <td>0.286398</td>
          <td>27.342075</td>
          <td>0.449643</td>
          <td>27.352084</td>
          <td>0.757605</td>
          <td>25.964290</td>
          <td>0.549914</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.531348</td>
          <td>0.434319</td>
          <td>25.752331</td>
          <td>0.084883</td>
          <td>24.801314</td>
          <td>0.032848</td>
          <td>23.873102</td>
          <td>0.023867</td>
          <td>23.171065</td>
          <td>0.024336</td>
          <td>22.770526</td>
          <td>0.038576</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.516196</td>
          <td>2.380840</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.475105</td>
          <td>0.342586</td>
          <td>26.688424</td>
          <td>0.286720</td>
          <td>26.253354</td>
          <td>0.360403</td>
          <td>25.277890</td>
          <td>0.347109</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.889811</td>
          <td>0.093884</td>
          <td>25.461123</td>
          <td>0.057619</td>
          <td>24.820722</td>
          <td>0.053718</td>
          <td>24.329443</td>
          <td>0.066091</td>
          <td>23.538647</td>
          <td>0.074462</td>
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
          <td>26.923597</td>
          <td>0.579181</td>
          <td>26.318414</td>
          <td>0.138834</td>
          <td>25.988809</td>
          <td>0.093767</td>
          <td>26.131333</td>
          <td>0.172557</td>
          <td>25.999037</td>
          <td>0.282225</td>
          <td>25.067860</td>
          <td>0.281099</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.749296</td>
          <td>0.505101</td>
          <td>27.261213</td>
          <td>0.300839</td>
          <td>26.825083</td>
          <td>0.189783</td>
          <td>26.074354</td>
          <td>0.161630</td>
          <td>26.048589</td>
          <td>0.289258</td>
          <td>25.421448</td>
          <td>0.366723</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>30.377304</td>
          <td>3.128249</td>
          <td>27.274029</td>
          <td>0.306116</td>
          <td>27.076746</td>
          <td>0.236112</td>
          <td>26.856919</td>
          <td>0.311897</td>
          <td>25.671827</td>
          <td>0.213955</td>
          <td>25.998152</td>
          <td>0.569465</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.206368</td>
          <td>0.294421</td>
          <td>26.647135</td>
          <td>0.167558</td>
          <td>25.844263</td>
          <td>0.136360</td>
          <td>25.747829</td>
          <td>0.231961</td>
          <td>25.516626</td>
          <td>0.404642</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.504523</td>
          <td>0.422105</td>
          <td>26.250531</td>
          <td>0.129665</td>
          <td>26.164823</td>
          <td>0.108201</td>
          <td>25.601246</td>
          <td>0.108032</td>
          <td>25.236837</td>
          <td>0.147594</td>
          <td>24.505104</td>
          <td>0.174222</td>
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
          <td>27.285221</td>
          <td>0.670893</td>
          <td>26.999227</td>
          <td>0.211655</td>
          <td>26.199252</td>
          <td>0.093982</td>
          <td>25.501472</td>
          <td>0.082780</td>
          <td>25.015938</td>
          <td>0.102884</td>
          <td>24.860971</td>
          <td>0.198479</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.314749</td>
          <td>0.590491</td>
          <td>28.077320</td>
          <td>0.446587</td>
          <td>27.784630</td>
          <td>0.541730</td>
          <td>27.387470</td>
          <td>0.686140</td>
          <td>26.623764</td>
          <td>0.761785</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.121211</td>
          <td>0.627058</td>
          <td>25.914590</td>
          <td>0.089340</td>
          <td>24.771381</td>
          <td>0.028876</td>
          <td>23.884394</td>
          <td>0.021707</td>
          <td>23.142679</td>
          <td>0.021477</td>
          <td>22.871354</td>
          <td>0.037918</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.674496</td>
          <td>0.975605</td>
          <td>27.628957</td>
          <td>0.421102</td>
          <td>27.354038</td>
          <td>0.310150</td>
          <td>26.469185</td>
          <td>0.238862</td>
          <td>25.617398</td>
          <td>0.214549</td>
          <td>25.006220</td>
          <td>0.278384</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.215339</td>
          <td>0.639717</td>
          <td>25.894510</td>
          <td>0.081764</td>
          <td>25.400854</td>
          <td>0.046423</td>
          <td>24.809781</td>
          <td>0.044904</td>
          <td>24.363732</td>
          <td>0.057934</td>
          <td>23.693415</td>
          <td>0.072238</td>
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
          <td>26.673258</td>
          <td>0.451892</td>
          <td>26.543315</td>
          <td>0.153776</td>
          <td>26.064333</td>
          <td>0.090307</td>
          <td>26.250927</td>
          <td>0.172105</td>
          <td>25.979594</td>
          <td>0.252291</td>
          <td>25.352498</td>
          <td>0.320449</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.125077</td>
          <td>0.605571</td>
          <td>26.926386</td>
          <td>0.201850</td>
          <td>26.737130</td>
          <td>0.152407</td>
          <td>26.684335</td>
          <td>0.232717</td>
          <td>25.795389</td>
          <td>0.204180</td>
          <td>25.763794</td>
          <td>0.417258</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.215736</td>
          <td>0.263459</td>
          <td>26.862446</td>
          <td>0.175037</td>
          <td>26.349131</td>
          <td>0.181523</td>
          <td>25.915970</td>
          <td>0.232805</td>
          <td>24.759198</td>
          <td>0.191231</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.885991</td>
          <td>0.469342</td>
          <td>26.709873</td>
          <td>0.163829</td>
          <td>25.847027</td>
          <td>0.126188</td>
          <td>25.970421</td>
          <td>0.258928</td>
          <td>25.084397</td>
          <td>0.267079</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.265229</td>
          <td>1.252124</td>
          <td>26.470478</td>
          <td>0.139583</td>
          <td>26.265888</td>
          <td>0.103568</td>
          <td>25.703633</td>
          <td>0.102977</td>
          <td>25.323655</td>
          <td>0.139675</td>
          <td>24.774781</td>
          <td>0.191845</td>
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
