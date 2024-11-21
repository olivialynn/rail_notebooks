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

    <pzflow.flow.Flow at 0x7fe8f260f250>



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
          <td>27.974180</td>
          <td>1.042455</td>
          <td>27.260282</td>
          <td>0.262586</td>
          <td>25.982390</td>
          <td>0.077630</td>
          <td>25.335716</td>
          <td>0.071495</td>
          <td>25.263196</td>
          <td>0.127590</td>
          <td>24.855608</td>
          <td>0.197560</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.921900</td>
          <td>1.010502</td>
          <td>28.553652</td>
          <td>0.696686</td>
          <td>27.708490</td>
          <td>0.335421</td>
          <td>30.148653</td>
          <td>2.065317</td>
          <td>27.162159</td>
          <td>0.585956</td>
          <td>26.638905</td>
          <td>0.768877</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.356157</td>
          <td>0.336835</td>
          <td>26.094194</td>
          <td>0.097320</td>
          <td>24.776186</td>
          <td>0.026717</td>
          <td>23.877838</td>
          <td>0.019854</td>
          <td>23.154890</td>
          <td>0.020034</td>
          <td>22.822174</td>
          <td>0.033328</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.252537</td>
          <td>1.222421</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.363699</td>
          <td>0.253988</td>
          <td>26.496154</td>
          <td>0.195639</td>
          <td>26.087238</td>
          <td>0.256083</td>
          <td>25.246805</td>
          <td>0.273107</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.265020</td>
          <td>0.313311</td>
          <td>25.753723</td>
          <td>0.072127</td>
          <td>25.341112</td>
          <td>0.043963</td>
          <td>24.818479</td>
          <td>0.045184</td>
          <td>24.257291</td>
          <td>0.052635</td>
          <td>23.692117</td>
          <td>0.072048</td>
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
          <td>26.644012</td>
          <td>0.421220</td>
          <td>26.240386</td>
          <td>0.110582</td>
          <td>26.223614</td>
          <td>0.096001</td>
          <td>25.837138</td>
          <td>0.111115</td>
          <td>26.411842</td>
          <td>0.332728</td>
          <td>25.677226</td>
          <td>0.384556</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.462957</td>
          <td>0.366309</td>
          <td>27.259913</td>
          <td>0.262507</td>
          <td>26.711898</td>
          <td>0.146759</td>
          <td>26.264538</td>
          <td>0.160746</td>
          <td>26.023452</td>
          <td>0.243002</td>
          <td>25.394291</td>
          <td>0.307652</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.449549</td>
          <td>1.359320</td>
          <td>26.997373</td>
          <td>0.211305</td>
          <td>27.080306</td>
          <td>0.200732</td>
          <td>26.511619</td>
          <td>0.198201</td>
          <td>25.826562</td>
          <td>0.206321</td>
          <td>24.908014</td>
          <td>0.206443</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.216685</td>
          <td>0.639819</td>
          <td>27.861505</td>
          <td>0.422520</td>
          <td>26.413669</td>
          <td>0.113362</td>
          <td>25.666019</td>
          <td>0.095662</td>
          <td>25.415239</td>
          <td>0.145484</td>
          <td>25.665964</td>
          <td>0.381211</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.586710</td>
          <td>0.820139</td>
          <td>26.544984</td>
          <td>0.143959</td>
          <td>26.099218</td>
          <td>0.086056</td>
          <td>25.706014</td>
          <td>0.099077</td>
          <td>25.145970</td>
          <td>0.115237</td>
          <td>25.219047</td>
          <td>0.267001</td>
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
          <td>26.849444</td>
          <td>0.541977</td>
          <td>26.746291</td>
          <td>0.196221</td>
          <td>26.067675</td>
          <td>0.098397</td>
          <td>25.401236</td>
          <td>0.089728</td>
          <td>24.965012</td>
          <td>0.115524</td>
          <td>24.988325</td>
          <td>0.258247</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.325193</td>
          <td>0.666059</td>
          <td>27.225921</td>
          <td>0.263786</td>
          <td>27.219775</td>
          <td>0.409718</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.533642</td>
          <td>2.261394</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.328155</td>
          <td>0.371541</td>
          <td>25.979582</td>
          <td>0.103591</td>
          <td>24.770557</td>
          <td>0.031971</td>
          <td>23.835048</td>
          <td>0.023097</td>
          <td>23.162648</td>
          <td>0.024160</td>
          <td>22.853954</td>
          <td>0.041531</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.769273</td>
          <td>1.034705</td>
          <td>28.195357</td>
          <td>0.638624</td>
          <td>27.369329</td>
          <td>0.314987</td>
          <td>26.560851</td>
          <td>0.258449</td>
          <td>28.162056</td>
          <td>1.290425</td>
          <td>25.037982</td>
          <td>0.286604</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.870904</td>
          <td>0.253853</td>
          <td>25.747962</td>
          <td>0.082888</td>
          <td>25.409743</td>
          <td>0.055052</td>
          <td>24.781230</td>
          <td>0.051868</td>
          <td>24.327194</td>
          <td>0.065959</td>
          <td>23.606475</td>
          <td>0.079057</td>
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
          <td>26.022881</td>
          <td>0.291435</td>
          <td>26.575727</td>
          <td>0.173015</td>
          <td>26.073201</td>
          <td>0.100968</td>
          <td>25.839532</td>
          <td>0.134374</td>
          <td>25.868296</td>
          <td>0.253688</td>
          <td>25.422340</td>
          <td>0.372649</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.855264</td>
          <td>1.053655</td>
          <td>27.835623</td>
          <td>0.469949</td>
          <td>26.808612</td>
          <td>0.187163</td>
          <td>26.187860</td>
          <td>0.178020</td>
          <td>26.249822</td>
          <td>0.339725</td>
          <td>25.030883</td>
          <td>0.268451</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.193624</td>
          <td>0.695548</td>
          <td>27.696591</td>
          <td>0.426000</td>
          <td>26.982089</td>
          <td>0.218273</td>
          <td>26.022692</td>
          <td>0.155994</td>
          <td>26.009240</td>
          <td>0.282424</td>
          <td>25.006008</td>
          <td>0.265228</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.494405</td>
          <td>0.856948</td>
          <td>27.584335</td>
          <td>0.396685</td>
          <td>26.628540</td>
          <td>0.164923</td>
          <td>25.753815</td>
          <td>0.126099</td>
          <td>25.593714</td>
          <td>0.204005</td>
          <td>26.367922</td>
          <td>0.746540</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.783234</td>
          <td>0.519763</td>
          <td>26.495481</td>
          <td>0.160044</td>
          <td>25.846903</td>
          <td>0.081858</td>
          <td>25.493079</td>
          <td>0.098278</td>
          <td>25.443849</td>
          <td>0.176131</td>
          <td>25.416048</td>
          <td>0.367140</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.671857</td>
          <td>0.160510</td>
          <td>25.999135</td>
          <td>0.078797</td>
          <td>25.310075</td>
          <td>0.069901</td>
          <td>24.970526</td>
          <td>0.098872</td>
          <td>25.126714</td>
          <td>0.247581</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.523356</td>
          <td>0.787416</td>
          <td>27.527789</td>
          <td>0.326029</td>
          <td>28.595761</td>
          <td>0.650231</td>
          <td>26.738847</td>
          <td>0.239743</td>
          <td>26.476418</td>
          <td>0.350433</td>
          <td>26.779833</td>
          <td>0.843263</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.108164</td>
          <td>0.291445</td>
          <td>25.957562</td>
          <td>0.092774</td>
          <td>24.818732</td>
          <td>0.030100</td>
          <td>23.867579</td>
          <td>0.021397</td>
          <td>23.110423</td>
          <td>0.020894</td>
          <td>22.848677</td>
          <td>0.037166</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.879779</td>
          <td>0.576590</td>
          <td>29.280726</td>
          <td>1.251893</td>
          <td>27.157065</td>
          <td>0.264484</td>
          <td>26.272581</td>
          <td>0.202804</td>
          <td>25.951551</td>
          <td>0.282434</td>
          <td>24.871171</td>
          <td>0.249310</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.367355</td>
          <td>0.340128</td>
          <td>25.785640</td>
          <td>0.074280</td>
          <td>25.401990</td>
          <td>0.046470</td>
          <td>24.796143</td>
          <td>0.044364</td>
          <td>24.386220</td>
          <td>0.059101</td>
          <td>23.699178</td>
          <td>0.072607</td>
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
          <td>26.180498</td>
          <td>0.308117</td>
          <td>26.266015</td>
          <td>0.121074</td>
          <td>26.126357</td>
          <td>0.095364</td>
          <td>26.313193</td>
          <td>0.181442</td>
          <td>25.682597</td>
          <td>0.197098</td>
          <td>25.319826</td>
          <td>0.312199</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.867599</td>
          <td>0.502967</td>
          <td>27.353639</td>
          <td>0.287060</td>
          <td>27.474174</td>
          <td>0.282210</td>
          <td>26.425196</td>
          <td>0.187360</td>
          <td>25.984787</td>
          <td>0.239037</td>
          <td>25.618666</td>
          <td>0.373048</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.347579</td>
          <td>0.293207</td>
          <td>26.808196</td>
          <td>0.167144</td>
          <td>26.238304</td>
          <td>0.165208</td>
          <td>26.371937</td>
          <td>0.336883</td>
          <td>25.563833</td>
          <td>0.368254</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.349806</td>
          <td>0.309768</td>
          <td>26.629716</td>
          <td>0.152976</td>
          <td>25.906744</td>
          <td>0.132883</td>
          <td>25.692607</td>
          <td>0.205696</td>
          <td>25.102076</td>
          <td>0.270955</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.501667</td>
          <td>0.386709</td>
          <td>26.651150</td>
          <td>0.162964</td>
          <td>26.097722</td>
          <td>0.089362</td>
          <td>25.700681</td>
          <td>0.102711</td>
          <td>25.315487</td>
          <td>0.138694</td>
          <td>24.864041</td>
          <td>0.206786</td>
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
