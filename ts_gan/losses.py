
def wgan_gp_loss(D_real, D_fake, gp, lambda_gp=10.0):
    d_loss = D_fake.mean() - D_real.mean() + lambda_gp * gp if D_real is not None else D_fake.mean()
    g_loss = -D_fake.mean()
    return d_loss, g_loss
