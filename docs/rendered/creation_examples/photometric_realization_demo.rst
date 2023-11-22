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
    from rail.creation.degradation.lsst_error_model import LSSTErrorModel
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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.13/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1c14e84df0>



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
    0      0.890625  27.370831  26.712662  26.025223  25.327188  25.016500   
    1      1.978239  29.557049  28.361185  27.587231  27.238544  26.628109   
    2      0.974287  26.566015  25.937716  24.787413  23.872456  23.139563   
    3      1.317979  29.042730  28.274593  27.501106  26.648790  26.091450   
    4      1.386366  26.292624  25.774778  25.429958  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362207  27.036276  26.823139  26.420132  26.110037   
    99997  1.372992  27.736044  27.271955  26.887581  26.416138  26.043434   
    99998  0.855022  28.044552  27.327116  26.599014  25.862331  25.592169   
    99999  1.723768  27.049067  26.526745  26.094595  25.642971  25.197956   
    
                   y     major     minor  
    0      24.926821  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346500  0.147522  0.143359  
    4      23.700010  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524904  0.044537  0.022302  
    99997  25.456165  0.073146  0.047825  
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
          <td>31.172297</td>
          <td>3.430576</td>
          <td>26.562721</td>
          <td>0.105583</td>
          <td>26.084861</td>
          <td>0.068194</td>
          <td>25.340978</td>
          <td>0.052257</td>
          <td>25.021891</td>
          <td>0.069445</td>
          <td>25.047443</td>
          <td>0.159796</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.038419</td>
          <td>0.362520</td>
          <td>27.490722</td>
          <td>0.229680</td>
          <td>28.102581</td>
          <td>0.525461</td>
          <td>26.066428</td>
          <td>0.172483</td>
          <td>25.834953</td>
          <td>0.307316</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.873697</td>
          <td>0.389236</td>
          <td>25.882633</td>
          <td>0.057988</td>
          <td>24.797719</td>
          <td>0.021944</td>
          <td>23.873355</td>
          <td>0.014716</td>
          <td>23.128763</td>
          <td>0.013557</td>
          <td>22.861474</td>
          <td>0.023448</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.914048</td>
          <td>0.817339</td>
          <td>27.705399</td>
          <td>0.277971</td>
          <td>27.204204</td>
          <td>0.180633</td>
          <td>26.703293</td>
          <td>0.172092</td>
          <td>25.931166</td>
          <td>0.153677</td>
          <td>25.795159</td>
          <td>0.297649</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.336934</td>
          <td>0.253759</td>
          <td>25.750773</td>
          <td>0.051593</td>
          <td>25.483414</td>
          <td>0.039993</td>
          <td>24.809233</td>
          <td>0.032626</td>
          <td>24.301733</td>
          <td>0.036670</td>
          <td>23.576059</td>
          <td>0.043921</td>
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
          <td>26.643909</td>
          <td>0.325091</td>
          <td>26.212954</td>
          <td>0.077661</td>
          <td>26.220695</td>
          <td>0.076900</td>
          <td>26.027656</td>
          <td>0.095907</td>
          <td>26.102146</td>
          <td>0.177794</td>
          <td>25.635739</td>
          <td>0.261534</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.621966</td>
          <td>0.319467</td>
          <td>26.982388</td>
          <td>0.151845</td>
          <td>26.542811</td>
          <td>0.102093</td>
          <td>26.446734</td>
          <td>0.138137</td>
          <td>25.959232</td>
          <td>0.157414</td>
          <td>25.461991</td>
          <td>0.226646</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.679523</td>
          <td>0.334399</td>
          <td>27.416936</td>
          <td>0.219265</td>
          <td>27.042587</td>
          <td>0.157411</td>
          <td>26.480484</td>
          <td>0.142215</td>
          <td>26.165722</td>
          <td>0.187622</td>
          <td>24.902178</td>
          <td>0.141068</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.886674</td>
          <td>0.393155</td>
          <td>27.355825</td>
          <td>0.208363</td>
          <td>26.494891</td>
          <td>0.097896</td>
          <td>25.783669</td>
          <td>0.077364</td>
          <td>25.514723</td>
          <td>0.107157</td>
          <td>25.333237</td>
          <td>0.203557</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.557109</td>
          <td>0.643300</td>
          <td>26.442709</td>
          <td>0.095055</td>
          <td>26.216528</td>
          <td>0.076618</td>
          <td>25.710465</td>
          <td>0.072517</td>
          <td>25.169914</td>
          <td>0.079153</td>
          <td>24.799610</td>
          <td>0.129108</td>
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
          <td>26.514637</td>
          <td>0.134631</td>
          <td>26.108640</td>
          <td>0.095240</td>
          <td>25.346561</td>
          <td>0.072902</td>
          <td>25.024178</td>
          <td>0.097718</td>
          <td>25.104389</td>
          <td>0.233340</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.944342</td>
          <td>0.434029</td>
          <td>27.455657</td>
          <td>0.298248</td>
          <td>28.838614</td>
          <td>1.087356</td>
          <td>25.894419</td>
          <td>0.206461</td>
          <td>25.694555</td>
          <td>0.375194</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.013092</td>
          <td>0.557070</td>
          <td>25.861406</td>
          <td>0.078796</td>
          <td>24.802078</td>
          <td>0.031155</td>
          <td>23.873729</td>
          <td>0.020779</td>
          <td>23.124168</td>
          <td>0.019229</td>
          <td>22.875754</td>
          <td>0.034882</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.629878</td>
          <td>0.887490</td>
          <td>27.502449</td>
          <td>0.337057</td>
          <td>27.069950</td>
          <td>0.240514</td>
          <td>26.735950</td>
          <td>0.267067</td>
          <td>25.844828</td>
          <td>0.220452</td>
          <td>26.207551</td>
          <td>0.605002</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.351610</td>
          <td>0.328150</td>
          <td>25.742504</td>
          <td>0.068560</td>
          <td>25.504481</td>
          <td>0.055896</td>
          <td>24.810308</td>
          <td>0.045361</td>
          <td>24.275106</td>
          <td>0.050466</td>
          <td>23.526552</td>
          <td>0.059663</td>
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
          <td>26.679934</td>
          <td>0.434865</td>
          <td>26.162959</td>
          <td>0.102465</td>
          <td>26.260038</td>
          <td>0.112524</td>
          <td>26.003706</td>
          <td>0.134226</td>
          <td>26.207316</td>
          <td>0.276502</td>
          <td>25.676039</td>
          <td>0.382003</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.457753</td>
          <td>0.358597</td>
          <td>26.963596</td>
          <td>0.198614</td>
          <td>26.449318</td>
          <td>0.129054</td>
          <td>26.457954</td>
          <td>0.192423</td>
          <td>25.899605</td>
          <td>0.208742</td>
          <td>25.435046</td>
          <td>0.307604</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.466663</td>
          <td>0.365212</td>
          <td>27.477524</td>
          <td>0.306782</td>
          <td>27.114231</td>
          <td>0.230048</td>
          <td>26.509794</td>
          <td>0.203816</td>
          <td>26.226686</td>
          <td>0.277236</td>
          <td>24.715459</td>
          <td>0.172050</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.641076</td>
          <td>0.427791</td>
          <td>27.368171</td>
          <td>0.288416</td>
          <td>26.450032</td>
          <td>0.134905</td>
          <td>25.747642</td>
          <td>0.109323</td>
          <td>25.477471</td>
          <td>0.152739</td>
          <td>25.253040</td>
          <td>0.277438</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.805294</td>
          <td>0.934084</td>
          <td>26.412937</td>
          <td>0.125229</td>
          <td>26.270516</td>
          <td>0.111556</td>
          <td>25.740679</td>
          <td>0.104875</td>
          <td>25.157535</td>
          <td>0.111735</td>
          <td>24.755998</td>
          <td>0.177247</td>
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
          <td>31.181209</td>
          <td>3.439376</td>
          <td>26.562682</td>
          <td>0.105608</td>
          <td>26.084880</td>
          <td>0.068215</td>
          <td>25.340983</td>
          <td>0.052274</td>
          <td>25.021893</td>
          <td>0.069469</td>
          <td>25.047489</td>
          <td>0.159856</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.037851</td>
          <td>0.362986</td>
          <td>27.490513</td>
          <td>0.230109</td>
          <td>28.105745</td>
          <td>0.527673</td>
          <td>26.065318</td>
          <td>0.172724</td>
          <td>25.834057</td>
          <td>0.307793</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.931294</td>
          <td>0.459125</td>
          <td>25.873453</td>
          <td>0.067085</td>
          <td>24.799592</td>
          <td>0.025907</td>
          <td>23.873516</td>
          <td>0.017320</td>
          <td>23.126789</td>
          <td>0.016000</td>
          <td>22.867628</td>
          <td>0.028372</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317979</td>
          <td>27.634029</td>
          <td>0.886619</td>
          <td>27.505489</td>
          <td>0.336272</td>
          <td>27.071999</td>
          <td>0.239678</td>
          <td>26.735416</td>
          <td>0.265557</td>
          <td>25.846158</td>
          <td>0.219492</td>
          <td>26.199574</td>
          <td>0.598784</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.337059</td>
          <td>0.254408</td>
          <td>25.750701</td>
          <td>0.051742</td>
          <td>25.483597</td>
          <td>0.040131</td>
          <td>24.809242</td>
          <td>0.032739</td>
          <td>24.301492</td>
          <td>0.036797</td>
          <td>23.575605</td>
          <td>0.044069</td>
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
          <td>26.658963</td>
          <td>0.371874</td>
          <td>26.191443</td>
          <td>0.088542</td>
          <td>26.237348</td>
          <td>0.091965</td>
          <td>26.017306</td>
          <td>0.112737</td>
          <td>26.146260</td>
          <td>0.219128</td>
          <td>25.652974</td>
          <td>0.314149</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.604982</td>
          <td>0.323862</td>
          <td>26.980541</td>
          <td>0.156570</td>
          <td>26.533220</td>
          <td>0.104999</td>
          <td>26.447842</td>
          <td>0.143599</td>
          <td>25.953123</td>
          <td>0.162919</td>
          <td>25.459259</td>
          <td>0.235232</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372992</td>
          <td>26.617184</td>
          <td>0.344143</td>
          <td>27.433262</td>
          <td>0.243064</td>
          <td>27.061988</td>
          <td>0.177088</td>
          <td>26.488572</td>
          <td>0.159397</td>
          <td>26.182451</td>
          <td>0.212451</td>
          <td>24.846435</td>
          <td>0.150974</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.736525</td>
          <td>0.415369</td>
          <td>27.363056</td>
          <td>0.255856</td>
          <td>26.468193</td>
          <td>0.120257</td>
          <td>25.762204</td>
          <td>0.096645</td>
          <td>25.492456</td>
          <td>0.134819</td>
          <td>25.284844</td>
          <td>0.249382</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.607365</td>
          <td>0.703136</td>
          <td>26.435987</td>
          <td>0.102015</td>
          <td>26.228480</td>
          <td>0.084300</td>
          <td>25.717249</td>
          <td>0.079788</td>
          <td>25.167075</td>
          <td>0.086746</td>
          <td>24.789461</td>
          <td>0.140677</td>
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
