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

    <pzflow.flow.Flow at 0x7f5f12146a40>



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
          <td>27.343556</td>
          <td>0.698102</td>
          <td>26.638299</td>
          <td>0.155954</td>
          <td>26.077314</td>
          <td>0.084411</td>
          <td>25.434950</td>
          <td>0.078050</td>
          <td>24.958916</td>
          <td>0.097858</td>
          <td>24.790969</td>
          <td>0.187086</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.736480</td>
          <td>0.451778</td>
          <td>29.109042</td>
          <td>0.994966</td>
          <td>26.902294</td>
          <td>0.172700</td>
          <td>27.764785</td>
          <td>0.533541</td>
          <td>26.598592</td>
          <td>0.385188</td>
          <td>26.546892</td>
          <td>0.723169</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.353755</td>
          <td>0.336197</td>
          <td>25.955403</td>
          <td>0.086158</td>
          <td>24.818892</td>
          <td>0.027732</td>
          <td>23.859553</td>
          <td>0.019550</td>
          <td>23.165111</td>
          <td>0.020208</td>
          <td>22.821606</td>
          <td>0.033312</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.078777</td>
          <td>0.497365</td>
          <td>27.800723</td>
          <td>0.360692</td>
          <td>26.359238</td>
          <td>0.174252</td>
          <td>25.857432</td>
          <td>0.211719</td>
          <td>25.346605</td>
          <td>0.296088</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.142149</td>
          <td>0.283857</td>
          <td>25.815433</td>
          <td>0.076165</td>
          <td>25.403107</td>
          <td>0.046449</td>
          <td>24.850616</td>
          <td>0.046492</td>
          <td>24.430508</td>
          <td>0.061381</td>
          <td>23.599937</td>
          <td>0.066402</td>
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
          <td>26.134817</td>
          <td>0.282178</td>
          <td>26.413969</td>
          <td>0.128575</td>
          <td>26.122114</td>
          <td>0.087808</td>
          <td>25.996077</td>
          <td>0.127582</td>
          <td>26.078811</td>
          <td>0.254320</td>
          <td>25.389913</td>
          <td>0.306574</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.527869</td>
          <td>2.221984</td>
          <td>27.036129</td>
          <td>0.218247</td>
          <td>27.432191</td>
          <td>0.268620</td>
          <td>26.645492</td>
          <td>0.221684</td>
          <td>25.714473</td>
          <td>0.187759</td>
          <td>25.434630</td>
          <td>0.317738</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.629939</td>
          <td>0.843258</td>
          <td>27.659260</td>
          <td>0.361389</td>
          <td>26.878612</td>
          <td>0.169256</td>
          <td>26.164312</td>
          <td>0.147517</td>
          <td>25.824620</td>
          <td>0.205986</td>
          <td>26.162974</td>
          <td>0.553339</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.927784</td>
          <td>0.199341</td>
          <td>26.466945</td>
          <td>0.118744</td>
          <td>25.822613</td>
          <td>0.109715</td>
          <td>25.431129</td>
          <td>0.147485</td>
          <td>25.879353</td>
          <td>0.448839</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.360688</td>
          <td>0.338044</td>
          <td>27.104068</td>
          <td>0.230917</td>
          <td>26.110473</td>
          <td>0.086913</td>
          <td>25.686416</td>
          <td>0.097389</td>
          <td>25.028978</td>
          <td>0.104051</td>
          <td>25.209440</td>
          <td>0.264916</td>
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
          <td>25.724457</td>
          <td>0.224947</td>
          <td>26.627050</td>
          <td>0.177432</td>
          <td>25.860893</td>
          <td>0.082042</td>
          <td>25.284698</td>
          <td>0.080977</td>
          <td>25.039147</td>
          <td>0.123212</td>
          <td>25.123686</td>
          <td>0.288308</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.951454</td>
          <td>0.510510</td>
          <td>28.155786</td>
          <td>0.541873</td>
          <td>27.332233</td>
          <td>0.446318</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.228820</td>
          <td>1.999561</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>29.568495</td>
          <td>2.392217</td>
          <td>25.905869</td>
          <td>0.097124</td>
          <td>24.793436</td>
          <td>0.032621</td>
          <td>23.872011</td>
          <td>0.023844</td>
          <td>23.167065</td>
          <td>0.024252</td>
          <td>22.910343</td>
          <td>0.043658</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.538712</td>
          <td>3.162228</td>
          <td>27.992538</td>
          <td>0.508482</td>
          <td>27.244420</td>
          <td>0.443235</td>
          <td>25.827108</td>
          <td>0.256045</td>
          <td>24.807913</td>
          <td>0.237460</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.969360</td>
          <td>0.590753</td>
          <td>25.753220</td>
          <td>0.083272</td>
          <td>25.473231</td>
          <td>0.058242</td>
          <td>24.777384</td>
          <td>0.051691</td>
          <td>24.331494</td>
          <td>0.066211</td>
          <td>23.726898</td>
          <td>0.087905</td>
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
          <td>26.473174</td>
          <td>0.415140</td>
          <td>26.499205</td>
          <td>0.162107</td>
          <td>26.118807</td>
          <td>0.105078</td>
          <td>25.672730</td>
          <td>0.116279</td>
          <td>25.670021</td>
          <td>0.215304</td>
          <td>25.695853</td>
          <td>0.459392</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.984445</td>
          <td>1.135643</td>
          <td>26.753690</td>
          <td>0.198156</td>
          <td>26.664357</td>
          <td>0.165600</td>
          <td>26.181821</td>
          <td>0.177110</td>
          <td>26.128727</td>
          <td>0.308519</td>
          <td>26.285804</td>
          <td>0.691402</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.577730</td>
          <td>0.446995</td>
          <td>26.845623</td>
          <td>0.215594</td>
          <td>26.945461</td>
          <td>0.211703</td>
          <td>26.553886</td>
          <td>0.243845</td>
          <td>26.172131</td>
          <td>0.321909</td>
          <td>26.465764</td>
          <td>0.785032</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.490602</td>
          <td>0.854874</td>
          <td>27.369174</td>
          <td>0.335302</td>
          <td>26.744973</td>
          <td>0.182071</td>
          <td>25.878865</td>
          <td>0.140490</td>
          <td>25.354090</td>
          <td>0.166598</td>
          <td>25.555675</td>
          <td>0.416934</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.684846</td>
          <td>0.954011</td>
          <td>26.423653</td>
          <td>0.150506</td>
          <td>26.079813</td>
          <td>0.100450</td>
          <td>25.673681</td>
          <td>0.115075</td>
          <td>25.370267</td>
          <td>0.165446</td>
          <td>25.297471</td>
          <td>0.334448</td>
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
          <td>26.996621</td>
          <td>0.547409</td>
          <td>26.444845</td>
          <td>0.132069</td>
          <td>25.985535</td>
          <td>0.077856</td>
          <td>25.333686</td>
          <td>0.071377</td>
          <td>24.891731</td>
          <td>0.092266</td>
          <td>24.807530</td>
          <td>0.189745</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.696479</td>
          <td>0.332528</td>
          <td>26.854400</td>
          <td>0.263610</td>
          <td>26.319339</td>
          <td>0.309351</td>
          <td>26.321116</td>
          <td>0.619735</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.072077</td>
          <td>0.605795</td>
          <td>26.041658</td>
          <td>0.099867</td>
          <td>24.798413</td>
          <td>0.029568</td>
          <td>23.896338</td>
          <td>0.021930</td>
          <td>23.160159</td>
          <td>0.021800</td>
          <td>22.802114</td>
          <td>0.035668</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.825408</td>
          <td>0.961951</td>
          <td>28.083927</td>
          <td>0.541950</td>
          <td>26.538766</td>
          <td>0.252943</td>
          <td>25.857642</td>
          <td>0.261653</td>
          <td>25.335129</td>
          <td>0.361881</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.774335</td>
          <td>0.209982</td>
          <td>25.770205</td>
          <td>0.073275</td>
          <td>25.419062</td>
          <td>0.047180</td>
          <td>24.795410</td>
          <td>0.044335</td>
          <td>24.335745</td>
          <td>0.056513</td>
          <td>23.679679</td>
          <td>0.071365</td>
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
          <td>26.414059</td>
          <td>0.370527</td>
          <td>26.426760</td>
          <td>0.139128</td>
          <td>26.154522</td>
          <td>0.097749</td>
          <td>26.076346</td>
          <td>0.148251</td>
          <td>25.549404</td>
          <td>0.176122</td>
          <td>25.749967</td>
          <td>0.436584</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.703309</td>
          <td>0.444984</td>
          <td>27.015412</td>
          <td>0.217446</td>
          <td>27.156844</td>
          <td>0.217389</td>
          <td>26.583189</td>
          <td>0.213945</td>
          <td>26.012685</td>
          <td>0.244600</td>
          <td>25.785154</td>
          <td>0.424116</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.799761</td>
          <td>0.487709</td>
          <td>27.324122</td>
          <td>0.287708</td>
          <td>26.923052</td>
          <td>0.184261</td>
          <td>26.429902</td>
          <td>0.194335</td>
          <td>26.194303</td>
          <td>0.292301</td>
          <td>25.341283</td>
          <td>0.308814</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.435234</td>
          <td>0.789784</td>
          <td>26.758148</td>
          <td>0.190361</td>
          <td>27.008039</td>
          <td>0.210762</td>
          <td>25.988461</td>
          <td>0.142588</td>
          <td>25.447936</td>
          <td>0.167278</td>
          <td>26.541582</td>
          <td>0.788122</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.162358</td>
          <td>0.295838</td>
          <td>26.522930</td>
          <td>0.146024</td>
          <td>26.075513</td>
          <td>0.087633</td>
          <td>25.630355</td>
          <td>0.096573</td>
          <td>25.056358</td>
          <td>0.110771</td>
          <td>24.844138</td>
          <td>0.203365</td>
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
