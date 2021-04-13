def loss_no_fair(output, target):
    loss = torch.mean((-target * torch.log(output.T[1])- (1 - target) * torch.log(output.T[0])))
    return loss

def loss_with_fair(output, target, Theta_X, Sen, Sen_bar):
    pred_loss = torch.mean((-target * torch.log(output.T[1])- (1 - target) * torch.log(output.T[0])))
    fair_loss = torch.mul(Sen - Sen_bar, Theta_X)
    fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))
    return pred_loss + 4.0*fair_loss