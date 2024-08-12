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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f9f496f3ca0>



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
          <td>27.209406</td>
          <td>0.636586</td>
          <td>26.615243</td>
          <td>0.152907</td>
          <td>26.126032</td>
          <td>0.088111</td>
          <td>25.328605</td>
          <td>0.071047</td>
          <td>25.001513</td>
          <td>0.101579</td>
          <td>24.734656</td>
          <td>0.178380</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.482073</td>
          <td>0.765955</td>
          <td>28.702756</td>
          <td>0.769831</td>
          <td>27.892590</td>
          <td>0.387441</td>
          <td>29.515280</td>
          <td>1.553267</td>
          <td>30.174547</td>
          <td>2.739346</td>
          <td>26.884971</td>
          <td>0.900664</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.341504</td>
          <td>0.332954</td>
          <td>25.968325</td>
          <td>0.087143</td>
          <td>24.781012</td>
          <td>0.026830</td>
          <td>23.859763</td>
          <td>0.019553</td>
          <td>23.174559</td>
          <td>0.020371</td>
          <td>22.859111</td>
          <td>0.034432</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.407935</td>
          <td>2.118320</td>
          <td>27.883527</td>
          <td>0.429662</td>
          <td>27.531008</td>
          <td>0.291041</td>
          <td>26.491056</td>
          <td>0.194802</td>
          <td>26.593166</td>
          <td>0.383571</td>
          <td>25.642312</td>
          <td>0.374265</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.604922</td>
          <td>0.408821</td>
          <td>25.923812</td>
          <td>0.083797</td>
          <td>25.349336</td>
          <td>0.044285</td>
          <td>24.801098</td>
          <td>0.044492</td>
          <td>24.363968</td>
          <td>0.057863</td>
          <td>23.664832</td>
          <td>0.070329</td>
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
          <td>26.336523</td>
          <td>0.331643</td>
          <td>26.395631</td>
          <td>0.126550</td>
          <td>26.216842</td>
          <td>0.095432</td>
          <td>25.889846</td>
          <td>0.116338</td>
          <td>25.844033</td>
          <td>0.209360</td>
          <td>25.514402</td>
          <td>0.338523</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.862781</td>
          <td>0.975093</td>
          <td>27.156528</td>
          <td>0.241149</td>
          <td>27.064363</td>
          <td>0.198061</td>
          <td>26.549940</td>
          <td>0.204680</td>
          <td>27.002991</td>
          <td>0.522405</td>
          <td>25.148530</td>
          <td>0.252028</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.092974</td>
          <td>0.586511</td>
          <td>27.533222</td>
          <td>0.327197</td>
          <td>27.260890</td>
          <td>0.233354</td>
          <td>26.230044</td>
          <td>0.156073</td>
          <td>25.941367</td>
          <td>0.227050</td>
          <td>25.661691</td>
          <td>0.379948</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.242039</td>
          <td>1.215342</td>
          <td>27.181048</td>
          <td>0.246069</td>
          <td>26.514318</td>
          <td>0.123733</td>
          <td>25.780682</td>
          <td>0.105769</td>
          <td>25.418553</td>
          <td>0.145900</td>
          <td>24.938790</td>
          <td>0.211827</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.622569</td>
          <td>0.153869</td>
          <td>26.097272</td>
          <td>0.085908</td>
          <td>25.754854</td>
          <td>0.103407</td>
          <td>25.322933</td>
          <td>0.134358</td>
          <td>24.921046</td>
          <td>0.208707</td>
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
          <td>26.844704</td>
          <td>0.489790</td>
          <td>26.880074</td>
          <td>0.191500</td>
          <td>26.107355</td>
          <td>0.086675</td>
          <td>25.371681</td>
          <td>0.073806</td>
          <td>24.850289</td>
          <td>0.088953</td>
          <td>24.702267</td>
          <td>0.173543</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.288871</td>
          <td>1.247095</td>
          <td>28.145398</td>
          <td>0.522312</td>
          <td>27.694402</td>
          <td>0.331698</td>
          <td>27.710935</td>
          <td>0.512958</td>
          <td>26.319395</td>
          <td>0.309098</td>
          <td>27.324942</td>
          <td>1.170069</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.833818</td>
          <td>0.485855</td>
          <td>25.946782</td>
          <td>0.085507</td>
          <td>24.838312</td>
          <td>0.028207</td>
          <td>23.863834</td>
          <td>0.019621</td>
          <td>23.139428</td>
          <td>0.019773</td>
          <td>22.776201</td>
          <td>0.032006</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.241692</td>
          <td>0.560079</td>
          <td>28.477971</td>
          <td>0.598291</td>
          <td>26.937405</td>
          <td>0.281772</td>
          <td>25.944793</td>
          <td>0.227696</td>
          <td>25.316742</td>
          <td>0.289040</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.432571</td>
          <td>0.357709</td>
          <td>25.727034</td>
          <td>0.070446</td>
          <td>25.467060</td>
          <td>0.049163</td>
          <td>24.811762</td>
          <td>0.044915</td>
          <td>24.318885</td>
          <td>0.055593</td>
          <td>23.642577</td>
          <td>0.068957</td>
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
          <td>26.710769</td>
          <td>0.443106</td>
          <td>26.536000</td>
          <td>0.142851</td>
          <td>26.092837</td>
          <td>0.085573</td>
          <td>25.837312</td>
          <td>0.111132</td>
          <td>25.990858</td>
          <td>0.236551</td>
          <td>25.562267</td>
          <td>0.351543</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.934690</td>
          <td>1.018264</td>
          <td>26.988734</td>
          <td>0.209785</td>
          <td>27.084510</td>
          <td>0.201441</td>
          <td>26.424977</td>
          <td>0.184237</td>
          <td>26.387336</td>
          <td>0.326318</td>
          <td>24.904028</td>
          <td>0.205754</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.492029</td>
          <td>0.771002</td>
          <td>26.784568</td>
          <td>0.176647</td>
          <td>26.645923</td>
          <td>0.138655</td>
          <td>26.257046</td>
          <td>0.159720</td>
          <td>25.998738</td>
          <td>0.238097</td>
          <td>25.361995</td>
          <td>0.299778</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.678037</td>
          <td>0.432262</td>
          <td>27.462951</td>
          <td>0.309365</td>
          <td>26.658680</td>
          <td>0.140189</td>
          <td>26.153833</td>
          <td>0.146194</td>
          <td>25.833343</td>
          <td>0.207496</td>
          <td>25.646107</td>
          <td>0.375373</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.470096</td>
          <td>0.368354</td>
          <td>26.791109</td>
          <td>0.177630</td>
          <td>26.118374</td>
          <td>0.087519</td>
          <td>25.684135</td>
          <td>0.097194</td>
          <td>25.091910</td>
          <td>0.109932</td>
          <td>25.085972</td>
          <td>0.239375</td>
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
          <td>27.347750</td>
          <td>0.700092</td>
          <td>26.985982</td>
          <td>0.209303</td>
          <td>25.978617</td>
          <td>0.077372</td>
          <td>25.314454</td>
          <td>0.070162</td>
          <td>24.974389</td>
          <td>0.099194</td>
          <td>24.940283</td>
          <td>0.212091</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.819343</td>
          <td>3.208570</td>
          <td>27.206299</td>
          <td>0.223020</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.133084</td>
          <td>0.573929</td>
          <td>26.634117</td>
          <td>0.766450</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.750779</td>
          <td>0.456660</td>
          <td>25.930947</td>
          <td>0.084325</td>
          <td>24.770763</td>
          <td>0.026591</td>
          <td>23.887646</td>
          <td>0.020020</td>
          <td>23.153948</td>
          <td>0.020018</td>
          <td>22.894006</td>
          <td>0.035509</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.622548</td>
          <td>0.839276</td>
          <td>27.849016</td>
          <td>0.418512</td>
          <td>27.744889</td>
          <td>0.345208</td>
          <td>26.534280</td>
          <td>0.202010</td>
          <td>26.804986</td>
          <td>0.451005</td>
          <td>26.796468</td>
          <td>0.851669</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.500586</td>
          <td>0.775358</td>
          <td>25.820998</td>
          <td>0.076539</td>
          <td>25.486344</td>
          <td>0.050012</td>
          <td>24.777795</td>
          <td>0.043582</td>
          <td>24.415400</td>
          <td>0.060564</td>
          <td>23.701939</td>
          <td>0.072676</td>
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
          <td>26.843236</td>
          <td>0.489258</td>
          <td>26.362698</td>
          <td>0.122990</td>
          <td>26.353343</td>
          <td>0.107549</td>
          <td>26.401012</td>
          <td>0.180537</td>
          <td>25.710157</td>
          <td>0.187076</td>
          <td>25.009369</td>
          <td>0.224659</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.637158</td>
          <td>3.237513</td>
          <td>27.108310</td>
          <td>0.231730</td>
          <td>27.027058</td>
          <td>0.191938</td>
          <td>26.375248</td>
          <td>0.176636</td>
          <td>25.850930</td>
          <td>0.210571</td>
          <td>25.443612</td>
          <td>0.320022</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.684629</td>
          <td>0.368626</td>
          <td>26.902600</td>
          <td>0.172745</td>
          <td>26.431417</td>
          <td>0.185243</td>
          <td>25.693996</td>
          <td>0.184538</td>
          <td>25.158556</td>
          <td>0.254110</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.051652</td>
          <td>0.221085</td>
          <td>26.694988</td>
          <td>0.144641</td>
          <td>25.892463</td>
          <td>0.116603</td>
          <td>25.782799</td>
          <td>0.198883</td>
          <td>25.250683</td>
          <td>0.273970</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.045601</td>
          <td>0.567008</td>
          <td>27.065151</td>
          <td>0.223581</td>
          <td>26.072041</td>
          <td>0.084020</td>
          <td>25.661635</td>
          <td>0.095294</td>
          <td>25.023986</td>
          <td>0.103597</td>
          <td>25.239681</td>
          <td>0.271528</td>
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
